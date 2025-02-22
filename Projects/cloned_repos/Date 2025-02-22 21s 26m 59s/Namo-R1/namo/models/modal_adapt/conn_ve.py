from typing import List, Tuple, Union
from torch import nn
import torch
import re
from namo.models.modal_adapt.adapt_ve.mlp import MLP
from namo.models.modal_adapt.adapt_ve.components import MlpGLU
from namo.models.modal_adapt.adapt_ve.pixelshuffle import get_pixel_shuffle
from namo.utils.utils import rank0_print


class ConnVE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        type_name = self.config.conn_ve_llm_type

        llm_hidden_size = config.text_config.hidden_size
        ve_hidden_size = getattr(
            config.vision_config, "hidden_size", config.vision_config.intermediate_size
        )
        rank0_print(f"==> current conn type: {type_name}")
        if type_name == "identity":
            modules = nn.Identity()
        elif type_name == "linear":
            modules = nn.Linear(ve_hidden_size, llm_hidden_size)
        elif "gelu" in type_name:
            modules = MLP(type_name, ve_hidden_size, llm_hidden_size)
        elif "pixelshuffle" in type_name:
            modules = get_pixel_shuffle(type_name, ve_hidden_size, llm_hidden_size)
        elif "ovis" in type_name:
            print(f"{type_name} is not supported")
        elif "glu" in type_name:
            rank0_print("==> Using MLP GLU.")
            modules = []
            m = MlpGLU(in_hidden_size=ve_hidden_size, out_hidden_size=llm_hidden_size)
            modules.append(m)
            modules = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {type_name}")
        self.layers = modules

    def forward(
        self,
        x_or_tuple: Union[
            Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List[torch.Tensor]
        ],
    ):
        x = x_or_tuple
        if isinstance(x, list):
            out = [self.layers(i) for i in x]
            return out
        else:
            out = self.layers(x)
            return out
