"""
Code referenced from:

InternVL mDPO

"""

import math
import os
import random
import shutil
import sys
import traceback
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from loguru import logger
import numpy as np
import json

import torch
import torch.distributed as dist
import transformers
from namo.dataargs import DataArguments, ModelArguments
from namo.dataset_dpo import dpo_concat_pad_data_collator, WeightedConcatDataset
from namo.dataset_dpo import build_datasets
from namo.models.configuration_namo import NamoConfig
from namo.models.namo import NamoForCausalLM
from namo.tainer_mdpo import MultimodalDPOTrainer
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
from trl import DPOConfig as DPOConfigTRL
from namo.models.symbols import IGNORE_INDEX
from namo.utils.hf_utils import get_latest_checkpoint
from namo.utils.utils import find_all_linear_names, rank0_print

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class DPOConfig(DPOConfigTRL):
    loss_type: Literal[
        "sigmoid",
        "hinge",
        "ipo",
        "bco_pair",
        "sppo_hard",
        "nca_pair",
        "robust",
        "aot",
        "aot_pair",
        "exo_pair",
        "sigmoid,bco_pair",
    ] = "sigmoid"


def main(attn_implementation="flash_attention_2"):

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    if model_args.pretrain_model_path is not None:
        pretrain_model_path = get_latest_checkpoint(model_args.pretrain_model_path)
        rank0_print(f"==> finetune from pretrained whole model: {pretrain_model_path}")
        model = NamoForCausalLM.from_pretrained(pretrain_model_path)
        rank0_print("==> pretrained model loaded.")
    else:
        text_config = AutoConfig.from_pretrained(
            model_args.llm_model_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=compute_dtype,
        )
        vision_config = AutoConfig.from_pretrained(
            model_args.ve_model_path,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        config = NamoConfig(
            text_config=text_config,
            vision_config=vision_config,
            attn_implementation=attn_implementation,
            torch_dtype=compute_dtype,
            conn_ve_llm_type=model_args.conn_ve_llm_type,
            longest_edge=model_args.max_img_size,
            **bnb_model_from_pretrained_args,
        )
        model = NamoForCausalLM(config=config)

    # just copy ref model
    ref_model = copy(model)

    rank0_print(f"==> current model dtype: {model.dtype}, set is: {compute_dtype}")
    tokenizer = model.get_namo().tokenizer

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception as e:
                print(f"enable_input_require_grads: {e}")
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            use_dora=training_args.use_dora,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if tokenizer.unk_token != None:
        tokenizer.pad_token = tokenizer.unk_token

    if tokenizer.pad_token_id == None:
        rank0_print(f"tokenizer.pad_token: {tokenizer.pad_token}")
        if "mistral" in model_args.model_name_or_path.lower():
            # important for mistral models
            tokenizer.pad_token_id = tokenizer.encode("<pad>")
        else:
            tokenizer.pad_token_id = tokenizer.encode(
                tokenizer.pad_token
                if tokenizer.pad_token is not None
                else tokenizer.eos_token
            )
        rank0_print(f"pad_token_id: {tokenizer.pad_token_id}")

    if (
        model_args.ve_model_path is not None
        or model_args.pretrain_model_path is not None
    ):
        logger.info("preparing ve model args...")
        # model.get_model().initialize_vision_modules(
        #     model_args=model_args, fsdp=training_args.fsdp
        # )
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        model.config.unfreeze_ve = training_args.unfreeze_ve = model_args.unfreeze_ve
        if training_args.unfreeze_ve:
            for p in model.get_vision_tower().parameters():
                p.requires_grad = True

        model.config.new_img_size = model_args.new_img_size
        model.config.longest_edge = data_args.longest_edge = model_args.max_img_size
        model.config.dynamic_size = data_args.dynamic_size
        vision_tower.image_processor.size["longest_edge"] = data_args.longest_edge
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.video_fps = data_args.video_fps
        model.config.video_frames_num = data_args.video_frames_num
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_conn_ve_llm = training_args.tune_conn_ve_llm = (
            model_args.tune_conn_ve_llm
        )
        if model_args.tune_conn_ve_llm:
            model.requires_grad_(False)
            for p in model.get_namo().conn_ve_llm.parameters():
                p.requires_grad = True

        model.config.freeze_conn_ve_llm = training_args.freeze_conn_ve_llm
        if training_args.freeze_conn_ve_llm:
            for p in model.get_namo().conn_ve_llm.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_namo().conn_ve_llm.to(
                dtype=compute_dtype, device=training_args.device
            )

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.conn_ve_llm_lr = training_args.conn_ve_llm_lr
        model.config.s2 = model_args.s2
        model.config.s2_scales = model_args.s2_scales
        model.config.s2_max_split_size = model_args.s2_max_split_size
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    logger.info("Model load finished.")
    train_dataset = build_datasets(
        data_args,
        tokenizer,
        None,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    ref_model.eval()
    # _freeze_params(ref_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    trainer = MultimodalDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=dpo_concat_pad_data_collator,
    )

    # Training
    if training_args.do_train:
        print(
            f"[Memory Usage before training] {torch.cuda.memory_allocated()/1024/1024/1024:.2f}GB"
        )
        train_result = trainer.train(resume_from_checkpoint=True)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(train_dataset)
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model_dir = model_args.model_name_or_path
        output_dir = training_args.output_dir
        for filename in [
            "conversation.py",
            "modeling_internvl_chat.py",
            "modeling_intern_vit.py",
            "modeling_internlm2.py",
            "configuration_internvl_chat.py",
            "configuration_intern_vit.py",
            "configuration_internlm2.py",
        ]:
            if os.path.exists(os.path.join(model_dir, filename)):
                shutil.copy(os.path.join(model_dir, filename), output_dir)


if __name__ == "__main__":
    main()
