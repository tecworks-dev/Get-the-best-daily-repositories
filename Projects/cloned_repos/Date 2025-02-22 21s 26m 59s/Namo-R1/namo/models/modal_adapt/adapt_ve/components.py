from torch.nn import LayerNorm
from torch import nn
import torch
from timm.models.regnet import RegStage
from timm.layers import LayerNorm, LayerNorm2d


class GLU(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class MlpGLU(nn.Module):
    def __init__(self, in_hidden_size, out_hidden_size):
        super(MlpGLU, self).__init__()

        ffn_hidden_size = out_hidden_size * 4  # out_hidden_size * 4 3584 * 4 = 14336
        self.linear_proj = GLU(
            hidden_size=out_hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            in_features=in_hidden_size,
        )

    def forward(self, x, attention_mask: torch.Tensor = None):
        x = self.linear_proj(x)
        return x
