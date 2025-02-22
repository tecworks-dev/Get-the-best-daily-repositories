from dataclasses import dataclass, field
from typing import Optional, List
import transformers


@dataclass
class ModelArguments:
    llm_model_path: Optional[str] = field(default="checkpoints/Qwen2.5-0.5B-Instruct")
    ve_model_path: Optional[str] = field(default="checkpoints/aimv2-large-patch14-224")
    ae_model_path: Optional[str] = None
    pretrain_model_path: Optional[str] = None
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_conn_ve_llm: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_conn_ve_llm_path: Optional[str] = field(default=None)
    pretrain_stage_1_5: Optional[str] = field(default=None)
    conn_ve_llm_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    new_img_size: Optional[int] = field(default=None)
    max_img_size: Optional[int] = field(default=None)
    normalized_before_model: Optional[bool] = field(default=True)
    unfreeze_ve: bool = field(default=False)
    unfreeze_ve_layer_index: Optional[int] = field(default=None)
    s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default="384,768")
    s2_max_split_size: int = field(default=384)


@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    video_frames_num: int = field(default=16)
    video_fps: Optional[int] = field(default=1)
    dynamic_size: bool = False
    native_size: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_conn_ve_llm: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512)
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    conn_ve_llm_lr: Optional[float] = None
    ve_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    use_dora: bool = False
