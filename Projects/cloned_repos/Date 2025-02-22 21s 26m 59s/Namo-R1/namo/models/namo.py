import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
)
from torch import nn
from namo.models.meta_vision import NamoMetaVisionForCausalLM
from namo.utils.utils import rank0_print, load_conn_weights
from namo.models.configuration_namo import NamoConfig
from namo.models.modal_adapt.conn_ve import ConnVE
from namo.models.vision.ve import get_ve
from namo.utils.hf_utils import auto_load_model, auto_load_tokenizer, SimpleForCausalLM


class NamoPretrainedModel(PreTrainedModel):
    config_class = NamoConfig
    base_model_prefix = "namo"
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    supports_gradient_checkpointing = True


class NamoModel(NamoPretrainedModel):

    _no_split_modules = []

    def __init__(self, config: NamoConfig):
        super().__init__(config)
        self.config = config
        self.llm = auto_load_model(self.config.text_config)
        self.tokenizer = auto_load_tokenizer(self.config)
        # ve always being trained, not need delay anymore.
        self.ve = get_ve(config, delay_load=False)
        self.conn_ve_llm = ConnVE(config)

    def get_llm(self):
        return self.llm

    def get_vision_tower(self):
        ve = getattr(self, "ve", None)
        return ve

    def load_conn_ve_llm_weights(self, pretrain_mm_mlp_adapter):
        load_conn_weights(pretrain_mm_mlp_adapter, self.conn_ve_llm, "conn_ve_llm")


class NamoForCausalLM(
    NamoPretrainedModel, NamoMetaVisionForCausalLM, SimpleForCausalLM
):
    _no_split_modules = []

    def __init__(self, config):
        NamoPretrainedModel.__init__(self, config)
        super(SimpleForCausalLM, self).__init__(config.text_config)
        self.config = config

        # using model property avoid duplicated share tensor
        self.namo = NamoModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        self.post_init()

    def get_namo(self):
        return self.namo

    @property
    def model(self):
        return self.namo.llm

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        pixel_attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                pixel_values,
                image_sizes,
                pixel_attention_mask,
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                pixel_values,
                image_sizes=image_sizes,
                pixel_attention_mask=pixel_attention_mask,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if pixel_attention_mask is not None:
            inputs["pixel_attention_mask"] = pixel_attention_mask
        return inputs
