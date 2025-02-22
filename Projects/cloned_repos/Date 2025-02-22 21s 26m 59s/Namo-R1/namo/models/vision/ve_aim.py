import math
import torch
from torch import nn
import torchvision
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    CLIPImageProcessor,
    AutoConfig,
    AutoModel,
)
import os
from transformers import AutoModel
from .ve_base import BaseVE
from loguru import logger
from namo.utils.utils import is_main_process
from . import AIMv2Model
from . import AIMv2ModelNative
from . import AIMv2Config
from .aimv2.modeling_aimv2_native import RMSNorm


class VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class AimV2VE(BaseVE):
    def _load_vision_tower(self):
        # other models can be customized here, normally AutoModel can handle well
        if os.path.exists(self.vision_tower_name):
            if "native" in self.vision_tower_name:
                if is_main_process():
                    logger.info(
                        f"loading AIMv2-native pretrain model: {self.vision_tower_name} {self.torch_dtype}"
                    )
                self.vision_tower = AIMv2ModelNative.from_pretrained(
                    self.vision_tower_name,
                    ignore_mismatched_sizes=True,
                    torch_dtype=self.torch_dtype,
                )
            else:
                if is_main_process():
                    logger.info(
                        f"loading AIMv2 pretrain model: {self.vision_tower_name}"
                    )
                self.vision_tower = AIMv2Model.from_pretrained(
                    self.vision_tower_name,
                    ignore_mismatched_sizes=True,
                    torch_dtype=self.torch_dtype,
                )
        else:
            if is_main_process():
                logger.info(f"creating AIMv2 model: {self.vision_tower_name}")
            if "native" in self.vision_tower_name:
                self.vision_tower = AIMv2ModelNative(
                    config=self.vision_config,
                )
            else:
                self.vision_tower = AIMv2Model(
                    config=self.vision_config,
                )

        # todo: should check if vision_tower_name exist, if not, using config._name_or_path
        # self.image_processor = CLIPImageProcessor.from_pretrained(
        #     self.vision_tower_name
        # )
        # self.image_processor.do_center_crop = False

        # add a new PatchMerger after vision tower?
        # self.patch_merger = VLPatchMerger(
        #     self.vision_config.dim, self.vision_config.context_dim
        # )

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            return image_features[:, 1:]
        elif self.select_feature in ["cls_patch", "same", "all", "default"]:
            return image_features
        else:
            raise ValueError(f"Invalid select feature: {self.select_feature}")

    def forward(self, images, image_sizes=None):
        return self.basic_forward(images)
