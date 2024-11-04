from typing import Optional, Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from ioblocks import GaussianMixtureIOLayer, FSQ

from transformer import Stack, ShapeRotator, Block as PerfBlock, GPTOutput, CACHE_FILL_VALUE, FFNN, Norm
from tokenizer import make_tokenizer


from utils import si_module, exists, isnt, tqdm0, print0, default, print0_colored
from utils import load_ckpt    


@si_module
class LatentQuantizer(nn.Module):
    class Config:
        compressor_config: Optional[FSQ.Config] = None

        dim: Optional[int] = None
        ff_dim: Optional[int] = None
        input_dim: int = None

        from_pretrained: Optional[Tuple[str, str]] = None

    def __init__(self, c: Config):
        super().__init__()

        if exists(c.from_pretrained):
            checkpoint = load_ckpt(*c.from_pretrained)
        else:
            assert exists(c.compressor_config), f'hmm {c}'

        self.compressor = c.compressor_config()
        self.ffnn = FFNN(c.dim, c.ff_dim)
        self.input = nn.Linear(c.input_dim, c.dim) if exists(c.input_dim) else nn.Identity()

        if exists(c.from_pretrained):
            self.load_state_dict(checkpoint)

    @T.no_grad()
    def forward(self, x, return_latent=False, known_latent=None):  
        """
        x: (B, S, D)
        """
        if exists(known_latent): 
            return self.compressor.indices_to_codes(known_latent)

        x = self.input(x)
        x = self.ffnn(x)
        x, tokens = self.compressor(x)
        
        if return_latent:
            return x, tokens
        return x


@si_module
class TransformerVAE(nn.Module):
    class Config:
        io_config: Optional[GaussianMixtureIOLayer.Config] = None
        stack_config: Optional[Stack.Config] = None
        quantizer_config: Optional[LatentQuantizer.Config] = None

        plex_layer: int = None
        plex_roll: int = 1
        split: bool = True

        from_pretrained: Optional[Tuple[str, str]] = None

    def __init__(self, c: Config):
        super().__init__()

        if exists(c.from_pretrained):
            checkpoint = load_ckpt(*c.from_pretrained)
        else:
            assert (exists(c.io_config) and exists(c.stack_config) and exists(c.quantizer_config)), f'hmm {c}'

        self.io = c.io_config()
        self.stack = c.stack_config()

        self.plex_layer = c.stack_config.layers//2
        self.plex_roll = c.plex_roll
        self.plex_dim = c.quantizer_config.dim

        assert self.plex_dim is not None and c.stack_config.dim is not None, f'One of the following are None: self.plex_dim: {self.plex_dim}, c.stack_config.dim: {c.stack_config.dim}'
        self.plex_projection = nn.Linear(self.plex_dim, c.stack_config.dim)
        self.out_norm = Norm(c.stack_config.dim)

        if c.split:
            self.io2 = c.io_config()
            self.plex_projection2 = nn.Linear(self.plex_dim, c.stack_config.dim)

            self.io2.fc_loc = None
            self.io2.fc_scale = None
            self.io2.fc_weight = None

        kv_heads = c.stack_config.kv_heads or c.stack_config.n_head
        head_dim = c.stack_config.dim // c.stack_config.n_head
        self.cache_num_layers = c.stack_config.layers + ((c.stack_config.layers - self.plex_layer) if c.split else 0)
        cache_shape = [self.cache_num_layers, c.stack_config.seq_len, 2, kv_heads, head_dim]
        self.cache_shape = cache_shape
        self.cache = [None] * self.cache_num_layers
        
        if exists(c.from_pretrained):
            result = self.load_state_dict(checkpoint, strict=False)
            print0_colored(result, 'yellow')

        self.quantizer = c.quantizer_config().eval()
        self.quantizer.requires_grad = False

    @T.no_grad()
    def quantize(self, x):
        if self.c.split:
            x1, x2 = x.chunk(2, dim=-1)
            with T.autocast(device_type='cuda', dtype=T.bfloat16):
                quantized1 = self.quantizer(x1)
                quantized2 = self.quantizer(x2)
            return quantized1, quantized2
        else:
            with T.autocast(device_type='cuda', dtype=T.bfloat16):
                return self.quantizer(x)
    
    @T.no_grad()
    def untokenize(self, token_data):
        return self.quantizer(None, known_latent=token_data)

    def init_cache(self, bsize, device, dtype, length:int=None):
        cache_shape = self.cache_shape.copy()
        cache_shape[1] = length or cache_shape[1]
        self.cache = T.full((bsize, *cache_shape), CACHE_FILL_VALUE, device=device, dtype=dtype).transpose(0, 1)

    def deinit_cache(self):
        self.cache = [None] * self.cache_num_layers
    
    @T.no_grad()
    def forward(self, data, next_tokens: Optional[Tuple[T.Tensor, T.Tensor]] = None, temps: Optional[Tuple[float, Tuple[float, float]]] = None):
        if self.c.split:
            x1, x2 = data.chunk(2, dim=-1)
            x = self.io.input(x1) + self.io2.input(x2)
        else:
            x = self.io.input(data)

        cache_idx = 0
        for l, layer in enumerate(self.stack.layers):
            if l == self.plex_layer:
                if self.c.split:
                    plex1, plex2 = self.quantize(data)
                    plex1 = T.roll(plex1, -self.c.plex_roll, dims=1)
                    plex2 = T.roll(plex2, -self.c.plex_roll, dims=1)
                    if exists(next_tokens):
                        plex1[:, -1:] = self.untokenize(next_tokens[0])
                        plex2[:, -1:] = self.untokenize(next_tokens[1])
                    x1 = x + self.plex_projection(plex1)
                    x2 = x + self.plex_projection2(plex2)
                else:
                    plex = self.quantize(data)
                    plex = T.roll(plex, -self.c.plex_roll, dims=1)
                    if exists(next_tokens):
                        plex[:, -1:] = self.untokenize(next_tokens)
                    x = x + self.plex_projection(plex)

            if l < self.plex_layer:
                x = layer(x, kv=self.cache[l])
            else:
                if self.c.split:
                    x1 = layer(x1, kv=self.cache[self.plex_layer + cache_idx])
                    cache_idx += 1
                    x2 = layer(x2, kv=self.cache[self.plex_layer + cache_idx])
                    cache_idx += 1
                else:
                    x = layer(x, kv=self.cache[l])

        with T.autocast(device_type='cuda', dtype=T.bfloat16):
            if self.c.split:
                x1, x2 = self.out_norm(x1), self.out_norm(x2)
                out1, out2 = self.io.output(x1), self.io.output(x2)
            else:
                x = self.out_norm(x)
                out = self.io.output(x)

        if isnt(temps):
            if self.c.split:
                return out1, out2
            else:
                return out
        else:
            if self.c.split:
                next_data1 = self.io.temp_sample(out1, temps)[:, -1:, :]
                next_data2 = self.io2.temp_sample(out2, temps)[:, -1:, :]
                next_data = T.cat([next_data1, next_data2], dim=-1)
                return next_data
            else:
                next_data = self.io.temp_sample(out, temps)[:, -1:, :]
                return next_data

@si_module
class HertzDevModel(nn.Module):
    class Config:
        dim: int
        vocab_size: int
        stack_config: Optional[Stack.Config] = None
        latent_size: int = 32

        split: bool = True

        quantizer_config: Optional[LatentQuantizer.Config] = None
        resynthesizer_config: Optional[TransformerVAE.Config] = None

        from_pretrained: Optional[Tuple[str, str]] = None

    def __init__(self, c: Config):
        super().__init__()

        if exists(c.from_pretrained):
            checkpoint = load_ckpt(*c.from_pretrained)
        else:
            assert (exists(c.stack_config)), f'hmm {c}'

        self.input = nn.Linear(c.latent_size, c.dim)
        if self.c.split:
            self.input2 = nn.Linear(c.latent_size, c.dim)

        self.shape_rotator = ShapeRotator(c.stack_config.dim//c.stack_config.n_head, c.stack_config.seq_len, theta=c.stack_config.theta)

        self.layers = nn.ModuleList([
            PerfBlock(
                dim=c.stack_config.dim,
                layer_id=l,
                n_head=c.stack_config.n_head,
                kv_heads=c.stack_config.kv_heads,
                ff_dim=c.stack_config.ff_dim,
                eps=c.stack_config.eps,
                shape_rotator=self.shape_rotator,
            ) for l in range(c.stack_config.layers)
        ])

        self.output = GPTOutput(c.dim, c.vocab_size)
        if self.c.split:
            self.output2 = GPTOutput(c.dim, c.vocab_size)

        self.cache = [None] * c.stack_config.layers
        self.kv_heads = c.stack_config.kv_heads or c.stack_config.n_head
        self.head_dim = c.stack_config.dim // c.stack_config.n_head

        if exists(c.from_pretrained):
            result = self.load_state_dict(checkpoint, strict=False)
            print0_colored(result, 'yellow')

        self.resynthesizer = c.resynthesizer_config().eval()
        self.resynthesizer.requires_grad = False

        self.audio_tokenizer = make_tokenizer(device='cpu')
        self.audio_cache = None
        self.audio_latent_cache = None
        self.use_audio_cache = False

    @T.no_grad()
    def tokenize(self, audio_data):
        orig_audio_shape = audio_data.shape
        if exists(self.audio_cache):
            audio_data = T.cat([self.audio_cache, audio_data], dim=-1)
            self.audio_cache = audio_data[..., -(6*16_000):]
        elif self.use_audio_cache:
            self.audio_cache = audio_data[..., -(6*16_000):]

        if audio_data.shape[1] == 2:
            enc_ch1 = self.audio_tokenizer.latent_from_data(audio_data[:, 0:1])
            enc_ch2 = self.audio_tokenizer.latent_from_data(audio_data[:, 1:2])
            return T.cat([enc_ch1, enc_ch2], dim=-1)[:, -(orig_audio_shape[-1]//2000):]
        else:
            return self.audio_tokenizer.latent_from_data(audio_data)[:, -(orig_audio_shape[-1]//2000):]

    @T.no_grad()
    def untokenize(self, token_data):
        if exists(self.audio_latent_cache):
            token_data = T.cat([self.audio_latent_cache, token_data], dim=1)
            self.audio_latent_cache = token_data[:, -(6*8):]
        elif self.use_audio_cache:
            self.audio_latent_cache = token_data[:, -(6*8):]
        
        if token_data.shape[-1] == 2*self.c.latent_size:
            dec_ch1 = self.audio_tokenizer.data_from_latent(token_data[:, :self.c.latent_size])
            dec_ch2 = self.audio_tokenizer.data_from_latent(token_data[:, self.c.latent_size:])
            return T.cat([dec_ch1, dec_ch2], dim=1)[..., -(token_data.shape[1]*2000):]
        else:
            return self.audio_tokenizer.data_from_latent(token_data)[..., -(token_data.shape[1]*2000):]

    def init_cache(self, bsize, device, dtype, length:int=None):
        cache_shape = [self.c.stack_config.layers, length or self.c.stack_config.seq_len, 2, self.kv_heads, self.head_dim]
        self.cache = T.full((bsize, *cache_shape), CACHE_FILL_VALUE, device=device, dtype=dtype).transpose(0, 1)
        self.resynthesizer.init_cache(bsize, device, dtype, length)
        self.use_audio_cache = True

    def deinit_cache(self):
        self.cache = [None] * len(self.layers)
        self.resynthesizer.deinit_cache()
        self.audio_cache = None
        self.audio_latent_cache = None
        self.use_audio_cache = False
    
    @T.no_grad()
    def forward(self, data):
        if self.c.split:    
            x1, x2 = data.chunk(2, dim=-1)
            x = self.input(x1) + self.input2(x2)
        else:
            x = self.input(data)

        for l, layer in enumerate(self.layers):
            x = layer(x, kv=self.cache[l])

        if self.c.split:
            return self.output(x), self.output2(x)
        else:
            return self.output(x)
        
    @T.no_grad()
    def next_audio_from_audio(self, audio_data: T.Tensor, temps=(0.8, (0.5, 0.1))):
        latents_in = self.tokenize(audio_data)
        next_latents = self.next_latent(latents_in, temps)
        next_model_latent = next_latents[..., self.c.latent_size:]
        audio_decoded = self.untokenize(next_model_latent)[..., -2000:]
        return audio_decoded

        
    @T.no_grad()
    def next_latent(self, model_input: T.Tensor, temps=(0.8, (0.5, 0.1))):

        if self.c.split:
            logits1, logits2 = self.forward(model_input)
            next_logits1 = logits1[:, -1]
            next_logits2 = logits2[:, -1]
            next_token1 = F.softmax(next_logits1 / temps[0], dim=-1).multinomial(1)
            next_token2 = F.softmax(next_logits2 / temps[0], dim=-1).multinomial(1)

            next_input = self.resynthesizer(model_input, next_tokens=(next_token1, next_token2), temps=temps[1])
        else:
            logits = self.forward(model_input)
            next_logits = logits[:, -1]
            next_token = F.softmax(next_logits / temps[0], dim=-1).multinomial(1)

            next_input = self.resynthesizer(model_input, next_tokens=next_token, temps=temps[1])

        return next_input


    @T.no_grad()
    def completion(self, data: T.Tensor, temps=(0.8, (0.5, 0.1)), gen_len=None, use_cache=True) -> T.Tensor:
        """
        only accepts latent-space data.
        """
        if use_cache:
            self.init_cache(data.shape[0], data.device, T.bfloat16)

        next_input = generated = data

        target_len = min(data.shape[1] + default(gen_len, data.shape[1]), self.c.stack_config.seq_len)
        
        for _ in tqdm0(range(data.shape[1], target_len)):
            model_input = next_input if use_cache else generated

            next_input = self.next_latent(model_input, temps)
    
            generated = T.cat([generated, next_input], dim=1)

        if use_cache:
            self.deinit_cache()
        return generated
    


def get_hertz_dev_config(is_split=True, use_pure_audio_ablation=False):
    if is_split:
        checkpoints = [('inference_care_50000', 'e4ff4fe5c7e9f066410d2a5673b7a935'), ('inference_scion_54000', 'cb8bc484423922747b277ebc2933af5d')]
    elif not use_pure_audio_ablation:
        checkpoints = [('inference_whip_72000', '5e7cee7316900737d55fc5d44cc7a8f7'), ('inference_caraway_112000', 'fcb8368ef8ebf7712f3e31e6856da580')]
    else:
        checkpoints = [('inference_whip_72000', '5e7cee7316900737d55fc5d44cc7a8f7'), ('inference_syrup_110000', '353c48f553f1706824c11f3bb6a049e9')]

    quantizer_config=LatentQuantizer.Config(
        from_pretrained=('inference_volcano_3', 'd42bf674022c5f84b051d5d7794f6169'),
        compressor_config=FSQ.Config(
            levels=[8,8,8,8,8],
            dim=2048,
            num_codebooks=1,
            keep_num_codebooks_dim=None,
            scale=None,
            allowed_dtypes=['float32', 'float64', 'bfloat16'],
            channel_first=False,
            projection_has_bias=True,
            return_indices=True,
            force_quantization_f32=True,
            use_rms=False
        ),
        dim=2048,
        ff_dim=8192,
        input_dim=32
    )

    resynthesizer_config=TransformerVAE.Config(
        io_config=GaussianMixtureIOLayer.Config(
            latent_dim=32,
            dim=4096,
            num_components=8,
        ),
        stack_config=Stack.Config(
            layers=8,
            dim=4096,
            seq_len=8192,
            n_head=16,
            ff_dim=11008,
            kv_heads=16,
            eps=1e-5,
            theta=10_000
        ),
        quantizer_config=quantizer_config,
        plex_layer=None,
        plex_roll=1,
        split=is_split,
        from_pretrained=checkpoints[0],
    )

    return HertzDevModel.Config(
        dim=4096,
        vocab_size=32_768,
        stack_config=Stack.Config(
            layers=32,
            dim=4096,
            seq_len=2048,
            n_head=32,
            ff_dim=None,
            kv_heads=None,
            eps=1e-5,
            theta=10_000,
        ),
        quantizer_config=quantizer_config,
        resynthesizer_config=resynthesizer_config,
        split=is_split,
        from_pretrained=checkpoints[1],
    )