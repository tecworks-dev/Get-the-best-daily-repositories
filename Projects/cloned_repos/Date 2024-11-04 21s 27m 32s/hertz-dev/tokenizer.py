import math
from dataclasses import dataclass
from typing import Union, Tuple, Literal

import torch as T
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from utils import load_ckpt
from utils.interp import print_colored
from utils import si_module, get_activation



# Adapted from https://github.com/facebookresearch/AudioDec

def Conv1d1x1(in_channels, out_channels, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)


class NonCausalConv1d(nn.Module):
    """1D noncausal convloution w/ 2-sides padding."""

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=-1, 
            dilation=1,
            groups=1,
            bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x
    

class NonCausalConvTranspose1d(nn.Module):
    """1D noncausal transpose convloution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=-1,
        output_padding=-1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if padding < 0:
            padding = (stride+1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C', T').
        """
        x = self.deconv(x)
        return x


class CausalConv1d(NonCausalConv1d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1,
        bias=True
    ):
        super(CausalConv1d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0,
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.pad_length = (kernel_size - 1) * dilation
    def forward(self, x):
        pad = nn.ConstantPad1d((self.pad_length, 0), 0.0)
        x = pad(x)
        return self.conv(x)

        
class CausalConvTranspose1d(NonCausalConvTranspose1d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        bias=True,
        pad_buffer=None,
    ):
        super(CausalConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
            bias=bias,
        )
        self.stride = stride
        self.pad_length = (math.ceil(kernel_size/stride) - 1)
        if pad_buffer is None:
            pad_buffer = T.zeros(1, in_channels, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)

    def forward(self, x):
        pad = nn.ReplicationPad1d((self.pad_length, 0))
        x = pad(x)
        return self.deconv(x)[:, :, self.stride : -self.stride]
    
    def inference(self, x):
        x = T.cat((self.pad_buffer, x), -1)
        self.pad_buffer = x[:, :, -self.pad_length:]
        return self.deconv(x)[:, :, self.stride : -self.stride]
    
    def reset_buffer(self):
        self.pad_buffer.zero_()


class NonCausalResUnit(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=7,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.activation = nn.ELU()
        self.conv1 = NonCausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
        self.conv2 = Conv1d1x1(out_channels, out_channels, bias)

    def forward(self, x):
        y = self.conv1(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y


class CausalResUnit(NonCausalResUnit):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=7,
        dilation=1,
        bias=False,
    ):
        super(CausalResUnit, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            bias=bias,
        )
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
    
    def inference(self, x):
        y = self.conv1.inference(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y
    

class ResNetBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        stride,
        kernel_size=7,
        dilations=(1, 3, 9),
        bias=True,
        mode='encoder',
    ):
        super().__init__()
        assert mode in ('encoder', 'decoder'), f"Mode ({mode}) is not supported!"

        self.mode = mode
        self.stride = stride

        ConvUnit = CausalConv1d if mode == 'encoder' else CausalConvTranspose1d
        
        res_channels = in_channels if mode == 'encoder' else out_channels

        res_units = [CausalResUnit(
            res_channels, 
            res_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        ) for dilation in dilations]

        if in_channels == out_channels:
            if mode == 'encoder':
                self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride)
            if mode == 'decoder':
                self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            conv_unit = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
            ) if in_channels != out_channels else nn.Identity()
        else:
            conv_unit = ConvUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2 * stride),
                stride=stride,
                bias=bias,
            )

        if mode == 'encoder':
            if in_channels == out_channels:
                self.res_block = nn.Sequential(*res_units, self.pool, conv_unit)
            else:
                self.res_block = nn.Sequential(*res_units, conv_unit)
        elif mode == 'decoder':
            if in_channels == out_channels:
                self.res_block = nn.Sequential(self.upsample, conv_unit, *res_units)
            else:
                self.res_block = nn.Sequential(conv_unit, *res_units)

    def forward(self, x):
        out = x
        for unit in self.res_block:
            out = unit(out)
        return out

    def inference(self, x):
        for unit in self.res_block:
            x = unit.inference(x)
        return x
    



@si_module
class ResNetStack(nn.Module):
    """
    ResNet encoder or decoder stack. Channel ratios
    and strides take the default order of from 
    data/io-layer, to the middle of the model.
    """
    class Config:
        input_channels: int = 1
        output_channels: int = 1
        encode_channels: int = 32
        decode_channel_multiplier: int = 1
        latent_dim: int = None
        kernel_size: int = 7
        bias: bool = True
        channel_ratios: Tuple[int, ...] = (2, 4, 8, 16)
        strides: Tuple[int, ...] = (3, 4, 5, 5)
        mode: Literal['encoder', 'decoder'] = 'encoder'
        
    def __init__(self, c: Config):
        super().__init__()
        assert c.mode in ('encoder', 'decoder'), f"Mode ({c.mode}) is not supported!"
        
        self.mode = c.mode

        assert len(c.channel_ratios) == len(c.strides)
        channel_ratios = (1,) + c.channel_ratios
        strides = c.strides
        self.middle_channels = c.encode_channels * channel_ratios[-1]
        if c.mode == 'decoder':
            channel_ratios = tuple(reversed(channel_ratios))
            strides = tuple(reversed(strides))

        self.multiplier = c.decode_channel_multiplier if c.mode == 'decoder' else 1
        res_blocks = [ResNetBlock(
            c.encode_channels * channel_ratios[s_idx] * self.multiplier, 
            c.encode_channels * channel_ratios[s_idx+1] * self.multiplier, 
            stride,
            kernel_size=c.kernel_size,
            bias=c.bias,
            mode=c.mode,
        ) for s_idx, stride in enumerate(strides)]

        data_conv = CausalConv1d(
            in_channels=c.input_channels if c.mode == 'encoder' else c.encode_channels * self.multiplier, 
            out_channels=c.encode_channels if c.mode == 'encoder' else c.output_channels, 
            kernel_size=c.kernel_size, 
            stride=1, 
            bias=False,
        )

        if c.mode == 'encoder':
            self.res_stack = nn.Sequential(data_conv, *res_blocks)
        elif c.mode == 'decoder':
            self.res_stack = nn.Sequential(*res_blocks, data_conv)

        if c.latent_dim is not None:
            self.latent_proj = Conv1d1x1(self.middle_channels, c.latent_dim, bias=c.bias) if c.mode == 'encoder' else Conv1d1x1(c.latent_dim, self.middle_channels, bias=c.bias)
        if self.multiplier != 1:
            self.multiplier_proj = Conv1d1x1(self.middle_channels, self.middle_channels * self.multiplier, bias=c.bias)

    def forward(self, x, return_feats=False):
        if self.c.latent_dim is not None and self.mode == 'decoder':
            x = self.latent_proj(x)
        if self.multiplier != 1:
            x = self.multiplier_proj(x)

        feats = []
        for block in self.res_stack:
            x = block(x)
            if return_feats:
                feats.append(x)
        if self.c.latent_dim is not None and self.mode == 'encoder':
            x = self.latent_proj(x)
            if return_feats:
                feats.append(x)
        if return_feats:
            return feats
        return x

    def inference(self, x):
        for block in self.res_stack:
            x = block.inference(x)
        return x

    def reset_buffer(self):
        def _reset_buffer(m):
            if isinstance(m, CausalConv1d) or isinstance(m, CausalConvTranspose1d):
                m.reset_buffer()
        self.apply(_reset_buffer)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_reset_parameters)


    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(
                m, nn.ConvTranspose1d
            ):
                nn.utils.parametrizations.weight_norm(m)

        self.apply(_apply_weight_norm)


    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                print(m)
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)



@si_module
class GaussianZ(nn.Module):
    class Config:
        dim: int
        latent_dim: int
        bias: bool = False
        use_weight_norm: bool = False
        
    def __init__(self, c: Config):
        super().__init__()
        
        self.proj_in = nn.Linear(c.dim, c.latent_dim * 2, bias=c.bias)
        self.proj_out = nn.Linear(c.latent_dim, c.dim, bias=c.bias)

        if c.use_weight_norm:
            self.proj_in = weight_norm(self.proj_in)
            self.proj_out = weight_norm(self.proj_out)

    def reparam(self, mu, logvar):
        std = T.exp(logvar / 2)
        eps = T.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        return T.mean(-0.5 * T.sum(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=(1, 2))
            )
        
    def repr_from_latent(self, latent: Union[dict, T.Tensor]):
        if isinstance(latent, T.Tensor):
            z = latent
        else:
            z = self.reparam(latent['mu'], latent['logvar'])
        l = self.proj_out(z)
        return l
    
    def forward(self, x: T.Tensor) -> Tuple[T.Tensor, dict]:
        mu, logvar = self.proj_in(x).chunk(2, dim=-1)
        kl_div = self.kl_divergence(mu, logvar)
        z = self.reparam(mu, logvar)
        xhat = self.proj_out(z)
        latent = {'mu': mu, 'logvar': logvar, 'z': z, 'kl_divergence': kl_div}
        return xhat, latent
    


@si_module
class WaveCodec(nn.Module):
    class Config:
        resnet_config: ResNetStack.Config = None
        sample_rate: int = 16_000
        use_weight_norm: bool = False

        compressor_config: dataclass = None

        norm_stddev: float = 1.0

    def __init__(self, c: Config):
        super().__init__()
        self.norm_stddev = c.norm_stddev
        self.encoder = c.resnet_config(mode='encoder')
        self.sample_rate = c.sample_rate

        self.total_stride = 1
        for stride in c.resnet_config.strides:
            self.total_stride *= stride
        self.tokens_per_second = self.sample_rate / self.total_stride
        
        self.compressor = c.compressor_config(dim=self.encoder.middle_channels)

        self.decoder = c.resnet_config(mode='decoder')        

        if c.use_weight_norm:
            self.encoder.apply_weight_norm()
            self.decoder.apply_weight_norm()
            self.encoder.reset_parameters()
            self.decoder.reset_parameters()

    def encode(self, data):
        return self.encoder(data/self.norm_stddev)

    def decode(self, latent):
        return self.decoder(latent.transpose(1, 2))*self.norm_stddev

    @T.no_grad()
    def latent_from_data(self, data, get_parameters=False):
        x = self.encode(data)
        l_in = x.transpose(1, 2)
        l, latent = self.compressor(l_in)
        return latent['z'] if not get_parameters else {
            'mu': latent['mu'],
            'logvar': latent['logvar'],
            'z': latent['z'],
        }
    
    @T.no_grad()
    def data_from_latent(self, latent):
        l = self.compressor.repr_from_latent(latent)
        x = self.decode(l)
        return x
    
    def process(self, x):
        return self.latent_from_data(x)

    def unprocess(self, latent):
        return self.data_from_latent(latent)

    def forward(self, audio_input):
        x = self.encode(audio_input)

        l_in = x.transpose(1, 2)
        l, latent = self.compressor(l_in)

        xhat = self.decode(l)
        return xhat, latent
    


def make_tokenizer(device='cuda'):
    generator_config = WaveCodec.Config(
        resnet_config=ResNetStack.Config(
            input_channels=1,
            output_channels=1,
            encode_channels=16,
            decode_channel_multiplier=4,
            kernel_size=7,
            bias=True,
            channel_ratios=(4, 8, 16, 16, 16, 16),
            strides=(2, 2, 4, 5, 5, 5),
            mode=None,
        ),
        use_weight_norm=True,

        compressor_config=GaussianZ.Config(
            dim=None,
            latent_dim=32,

            bias=True,
            use_weight_norm=True
        ),

        norm_stddev=0.05,
    )
    checkpoint = load_ckpt("inference_apatosaurus_95000", expected_hash="ba876edb97b988e9196e449dd176ca97")

    tokenizer = generator_config()

    load_result = tokenizer.load_state_dict(checkpoint, strict=False)
    print_colored(f"Loaded tokenizer state dict: {load_result}", "grey")

    tokenizer = tokenizer.eval()
    # Only convert to bfloat16 if using CUDA
    if device == 'cuda':
        tokenizer = tokenizer.bfloat16()
    tokenizer = tokenizer.to(device)
    tokenizer.requires_grad_ = False
    return tokenizer

