import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoImageProcessor
import torchvision
from loguru import logger
from namo.utils.utils import rank0_print, is_main_process


class BaseVE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        delay_load = kwargs.pop("delay_load", False)
        self.vision_config = config.vision_config
        self.model_name_or_path = config._name_or_path
        self.torch_dtype = config.vision_config.torch_dtype
        self.is_loaded = False
        self.vision_tower_name = config.vision_config._name_or_path.lower()
        self.select_layer = config.vision_feature_layer
        self.select_feature = config.vision_feature_select_strategy
        self.new_img_size = config.new_img_size
        self.unfreeze_ve = config.unfreeze_ve
        self.longest_edge = config.longest_edge
        self.shortest_edge = config.shortest_edge

        # actually delay_load doesn't needed anymore
        if not delay_load:
            self.load_model()
        elif self.unfreeze_ve:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name, trust_remote_code=True
            )
            if self.new_img_size:
                logger.info(f"Update using new image size: {self.cfg_only.image_size}")
            self._load_image_processor()

    def load_model(self):
        if is_main_process():
            logger.info(f"Loading base components for {self.vision_tower_name}")
        self._load_image_processor()
        self._load_vision_tower()

        if self.new_img_size:
            if is_main_process():
                logger.info(f"set new image size {self.new_img_size}")
            self.image_processor.size = {
                "height": self.new_img_size,
                "width": self.new_img_size,
            }

        if not self.unfreeze_ve:
            self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _load_image_processor(self):
        processor_path = self.vision_tower_name
        logger.info(f"loading image processor from: {self.vision_tower_name}")
        if os.path.exists(self.vision_tower_name):
            processor_path = self.vision_tower_name
        elif os.path.exists(self.model_name_or_path):
            processor_path = self.model_name_or_path
        else:
            raise FileNotFoundError(
                f"No processor found for either {self.vision_tower_name} or {self.model_name_or_path}"
            )

        self.image_processor = AutoImageProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )

        if is_main_process():
            logger.info(f"==> self.longest_edge {self.longest_edge}")
        if self.longest_edge is not None:
            self.image_processor.do_resize = True
            # override from default image processor
            self.image_processor.size["longest_edge"] = self.longest_edge
            self.image_processor.size["shortest_edge"] = 42  # hard coded here
            if is_main_process():
                logger.info(f"==> override longest_edge {self.image_processor.size}")
        if self.shortest_edge is not None:
            self.image_processor.size["shortest_edge"] = self.shortest_edge
            if is_main_process():
                logger.info(f"override shortest_edge {self.shortest_edge}")

    def _load_vision_tower(self):
        raise NotImplementedError

    def feature_select(self, image_forward_outs):
        raise NotImplementedError

    def basic_forward(self, images):
        if isinstance(images, list):
            return self._process_image_list(images)
        else:
            return self._process_batch(images)

    def _process_image_list(self, images):
        image_features = []
        for image in images:
            inputs = image.to(device=self.device, dtype=self.dtype)
            # print(f'image shape: {inputs.shape}')
            if len(inputs.shape) < 4:
                inputs = inputs.unsqueeze(0)
            features = self._get_features(inputs)
            image_features.append(features)
        return image_features

    def _process_batch(self, images):
        inputs = images.to(device=self.device, dtype=self.dtype)
        return self._get_features(inputs)

    def _get_features(self, inputs):
        outputs = self.vision_tower(inputs, output_hidden_states=True)
        return self.feature_select(outputs).to(inputs.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def image_size_auto(self):
        return self.new_img_size or self.config.image_size

    def save_pretrained(self, model_path):
        self.vision_tower.save_pretrained(model_path)
