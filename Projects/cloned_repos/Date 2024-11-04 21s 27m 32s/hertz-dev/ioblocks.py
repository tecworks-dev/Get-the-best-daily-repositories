from __future__ import annotations
from functools import partial
from contextlib import nullcontext
from typing import List, Tuple
from math import ceil

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack


from utils import si_module, exists, default, maybe


@si_module
class GaussianMixtureIOLayer(nn.Module):
    class Config:
        latent_dim: int
        dim: int
        num_components: int

    def __init__(self, c: Config):
        super().__init__()
        self.latent_dim = c.latent_dim
        self.num_components = c.num_components
        self.input_projection = nn.Linear(c.latent_dim, c.dim)
        
        self.fc_loc = nn.Linear(c.dim, c.num_components * c.latent_dim)
        self.fc_scale = nn.Linear(c.dim, c.num_components * c.latent_dim)
        self.fc_weight = nn.Linear(c.dim, c.num_components)
    
    def _square_plus(self, x):
        return (x + T.sqrt(T.square(x) + 4)) / 2
    
    def input(self, sampled_latents: T.Tensor) -> T.Tensor:
        """Pre-sampled latents T.Tensor (B, L, Z) -> float tensor (B, L, D)"""
        hidden = self.input_projection(sampled_latents)
        return hidden
    
    def output(self, h: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        """float tensor (B, L, D) -> Tuple of locs, scales, and weights"""
        batch_size, seq_len, _ = h.shape
        
        locs = self.fc_loc(h).view(batch_size, seq_len, self.num_components, self.latent_dim)
        scales = T.clamp(self._square_plus(self.fc_scale(h)), min=1e-6).view(batch_size, seq_len, self.num_components, self.latent_dim)
        weights = self.fc_weight(h).view(batch_size, seq_len, self.num_components)
        
        return (locs, scales, weights)

    def loss(self, data, dataHat):
        locs, scales, weights = dataHat
        log_probs = -0.5 * T.sum(
            (data.unsqueeze(-2) - locs).pow(2) / scales.pow(2) +
            2 * T.log(scales) +
            T.log(T.tensor(2 * T.pi)),
            dim=-1
        )
        log_weights = F.log_softmax(weights, dim=-1)
        return -T.logsumexp(log_weights + log_probs, dim=-1)

    
    def temp_sample(self, orig_pdist, temp):
        locs, scales, weights = orig_pdist
        if temp is None:
            component_samples = locs + scales * T.randn_like(scales)
            mixture_samples = F.gumbel_softmax(weights, hard=True)
            sampled = (component_samples * mixture_samples.unsqueeze(-1)).sum(dim=-2)
        elif isinstance(temp, tuple):
            assert len(temp) == 2
            categorical_temp, gaussian_temp = temp
            component_samples = locs + scales * gaussian_temp * T.randn_like(scales)
            mixture_samples = F.gumbel_softmax(weights / categorical_temp, hard=True)
            sampled = (component_samples * mixture_samples.unsqueeze(-1)).sum(dim=-2)
        else:
            component_samples = locs + scales * temp * T.randn_like(scales)
            mixture_samples = F.gumbel_softmax(weights / temp, hard=True)
            sampled = (component_samples * mixture_samples.unsqueeze(-1)).sum(dim=-2)
        return sampled


class GPTOutput(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        return self.output(x)


# helper functions

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def first(l):
    return l[0]

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def get_code_utilization(codes, codebook_size, get_global=False):
    if get_global and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if world_size > 1:
        gathered_tokens = [T.zeros_like(codes) for _ in range(world_size)]
        dist.all_gather(gathered_tokens, codes)
        gathered_tokens = T.cat(gathered_tokens, dim=0)
    else:
        gathered_tokens = codes
    unique_tokens = len(T.unique(gathered_tokens))
    code_utilization = unique_tokens / min(gathered_tokens.numel(), codebook_size)
    return code_utilization

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class
# lucidrains fsq
@si_module
class FSQ(nn.Module):
    @property
    def needs_float32_params(self):
        return True
    
    class Config:
        levels: List[int]
        dim: int | None = None
        num_codebooks: int = 1
        keep_num_codebooks_dim: bool | None = None
        scale: float | None = None
        allowed_dtypes: Tuple[str, ...] = ('float32', 'float64')
        channel_first: bool = False
        projection_has_bias: bool = True
        return_indices: bool = True
        force_quantization_f32: bool = True
        use_rms: bool = False
        
    def __init__(self, c: Config):
        super().__init__()
        _levels = T.tensor(c.levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = T.cumprod(T.tensor([1] + c.levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = c.scale

        codebook_dim = len(c.levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * c.num_codebooks
        self.num_codebooks = c.num_codebooks

        self.allowed_dtypes = []
        for dtype_str in c.allowed_dtypes:
            if hasattr(T, dtype_str):
                self.allowed_dtypes.append(getattr(T, dtype_str))
            else:
                raise ValueError(f"Invalid dtype string: {dtype_str}")
            
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(c.keep_num_codebooks_dim, c.num_codebooks > 1)
        assert not (c.num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(c.dim, len(_levels) * c.num_codebooks)

        self.channel_first = c.channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = c.projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = c.projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        self.return_indices = c.return_indices
        if c.return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(T.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = c.allowed_dtypes
        self.force_quantization_f32 = c.force_quantization_f32

        self.latent_loss = None

    def latent_metric(self, codes, get_global=False):
        return {'code_util_estimate': get_code_utilization(codes, self.codebook_size, get_global)}

    def repr_from_latent(self, latent):
        return self.indices_to_codes(latent)

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = T.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    # @autocast(device_type='cuda', enabled = False)
    def forward(self, z, return_codes=False):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, device_type='cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.type(orig_dtype)

        # project out
        if return_codes:
            return codes, indices

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices