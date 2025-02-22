import os
from typing import List, Optional, Tuple, Union
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
from .utils import rank0_print
from pathlib import Path
import json
from transformers import TrainerState
from peft import PeftModel
import glob
from loguru import logger


def auto_load_model(config):
    model_name_or_path = config._name_or_path
    if os.path.exists(model_name_or_path):
        return AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=config.torch_dtype, trust_remote_code=True
        )
    else:
        return AutoModel.from_config(config=config, trust_remote_code=True)


def auto_load_tokenizer(config):
    model_name_or_path = config._name_or_path
    if hasattr(config, "text_config"):
        text_config = config.text_config
        text_model_name_or_path = getattr(text_config, "_name_or_path", None)
        if text_model_name_or_path:
            if os.path.exists(text_model_name_or_path):
                model_name_or_path = text_model_name_or_path
    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def try_resume_conn_weights(model, output_dir, weight_file_name="conn_ve_llm.bin"):
    checkpoint_dirs = list(Path(output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        sorted_checkpoints = sorted(
            checkpoint_dirs, key=lambda x: int(x.stem.split("-")[-1])
        )
        latest_checkpoint = sorted_checkpoints[-1]
        weights_path = latest_checkpoint / weight_file_name
        if weights_path.exists():
            if "ve_llm" in weight_file_name:
                model.namo.load_conn_ve_llm_weights(weights_path)
                rank0_print(f"resumed conn_ve_llm weights from: {weights_path}")

            state_path = latest_checkpoint / "trainer_state.json"
            if state_path.exists():
                state = TrainerState.load_from_json(state_path)
                epoch = state.epoch
                global_step = state.global_step
                max_steps = state.max_steps
                rank0_print(
                    f"tainer state resumed: {state_path}, epoch: {epoch} {global_step}/{max_steps}"
                )
                return state
            return None
        else:
            rank0_print(f"not resumed as: {weights_path} not found.")


def get_latest_checkpoint(output_dir, weights_file_name=None):
    if os.path.exists(output_dir) and os.path.isfile(output_dir):
        return output_dir
    if weights_file_name is not None and os.path.exists(
        os.path.join(output_dir, weights_file_name)
    ):
        return os.path.join(output_dir, weights_file_name)
    else:
        checkpoint_dirs = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoint_dirs:
            sorted_checkpoints = sorted(
                checkpoint_dirs, key=lambda x: int(x.stem.split("-")[-1])
            )
            latest_checkpoint = sorted_checkpoints[-1]
            if weights_file_name is None:
                return latest_checkpoint
            weights_path = latest_checkpoint / weights_file_name
            rank0_print(f"==> loading conn middle checkpoint from: {weights_path}")
            return weights_path


def find_and_merge_lora_adapters(model, model_path):
    def find_latest_checkpoint(model_path):
        checkpoints = glob.glob(os.path.join(model_path, "checkpoint-*"))
        if not checkpoints:
            return None
        return max(checkpoints, key=os.path.getctime)

    lora_path = None
    lora_adapters = glob.glob(
        os.path.join(model_path, "*.safetensors")
    )  # 假设适配器是 safetensors 格式
    if len(lora_adapters) > 0:
        lora_path = lora_adapters[0]
    else:
        latest_checkpoint = find_latest_checkpoint(model_path)
        if latest_checkpoint:
            lora_path = latest_checkpoint

    if lora_path:
        logger.info(f"Merging LoRA adapters: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    return model


class SimpleForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.vocab_size = config.vocab_size
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # for placeholder, must set in real model.
        self.lm_head = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
