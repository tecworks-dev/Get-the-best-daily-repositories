import re
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, projector_type, input_dim, hidden_dim, output_dim=None):

        super().__init__()
        self.projector_type = projector_type
        self.output_dim = output_dim or hidden_dim

        self.mlp_depth, self.use_norm = self._parse_projector_type()
        self.layers = self._build_layers(input_dim, hidden_dim)

    def _parse_projector_type(self):
        use_norm = "_Norm" in self.projector_type
        clean_type = self.projector_type.replace("_Norm", "")

        pattern = r"^mlp(\d+)x_gelu$"
        match = re.match(pattern, clean_type)
        if not match:
            raise ValueError(f"Invalid projector_type: {self.projector_type}")
        return int(match.group(1)), use_norm

    def _build_layers(self, input_dim, hidden_dim):
        modules = []

        modules.append(nn.Linear(input_dim, hidden_dim))
        if self.use_norm:
            modules.append(nn.LayerNorm(hidden_dim))

        for _ in range(self.mlp_depth - 1):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            if self.use_norm:
                modules.append(nn.LayerNorm(hidden_dim))

        if hidden_dim != self.output_dim:
            modules.append(nn.Linear(hidden_dim, self.output_dim))
        return nn.Sequential(*modules)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.flatten(1, 2)
        return self.layers(x)
