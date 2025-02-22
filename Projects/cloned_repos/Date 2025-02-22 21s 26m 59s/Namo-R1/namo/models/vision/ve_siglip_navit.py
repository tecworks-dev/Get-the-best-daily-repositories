import math
import torch
from torch import nn
import torchvision
from transformers import (
    AutoModel,
)
from transformers import AutoModel
from .ve_base import BaseVE
from loguru import logger


class SigLipNavitVE(BaseVE):
    def _load_vision_tower(self):
        logger.info(f"Loading AIMv2 specific model: {self.vision_tower_name}")
        # other models can be customized here, normally AutoModel can handle well
        self.vision_tower = AutoModel.from_pretrained(
            self.vision_tower_name, ignore_mismatched_sizes=True, trust_remote_code=True
        )
        self.image_processor.do_center_crop = False

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            return image_features[:, 1:]
        elif self.select_feature in ["cls_patch", "same"]:
            return image_features
        else:
            raise ValueError(f"Invalid select feature: {self.select_feature}")

    def forward(self, images, image_sizes=None):
        return self.basic_forward(images)
