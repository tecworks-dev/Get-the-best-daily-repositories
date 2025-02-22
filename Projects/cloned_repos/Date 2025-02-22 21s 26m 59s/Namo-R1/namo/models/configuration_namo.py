from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import AutoConfig, CONFIG_MAPPING


class NamoConfig(PretrainedConfig):

    model_type = "namo"
    is_composition = False
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": AutoConfig,
        "audio_config": AutoConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        ignore_index=-100,
        image_token_index=-200,
        vision_feature_select_strategy="same",
        vision_feature_layer=-2,
        image_seq_length=576,
        new_img_size=None,
        shortest_edge=None,
        longest_edge=None,
        unfreeze_ve=True,
        multimodal_projector_bias=True,
        conn_ve_llm_type="mlp2x_gelu",
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.image_seq_length = image_seq_length
        self.new_img_size = new_img_size
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.unfreeze_ve = unfreeze_ve
        self.conn_ve_llm_type = conn_ve_llm_type

        if vision_feature_select_strategy not in ["same", "patch"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'same', 'patch'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"]
                if "model_type" in vision_config
                else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"]
                if "model_type" in audio_config
                else "whisper"
            )
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["whisper"]()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "llama"
            )
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        super().__init__(**kwargs)
