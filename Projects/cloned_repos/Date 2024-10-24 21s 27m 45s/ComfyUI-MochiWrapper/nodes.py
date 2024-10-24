import os
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
from einops import rearrange
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from .mochi_preview.t2v_synth_mochi import T2VSynthMochiModel
from .mochi_preview.vae.model import Decoder

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

script_directory = os.path.dirname(os.path.abspath(__file__))

def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
    const = quadratic_coef * (linear_steps ** 2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule
   
class DownloadAndLoadMochiModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [   
                        "mochi_preview_dit_fp8_e4m3fn.safetensors",
                        "mochi_preview_dit_bf16.safetensors",
                        "mochi_preview_dit_GGUF_Q4_0_v2.safetensors"

                    ],
                    {"tooltip": "Downloads from 'https://huggingface.co/Kijai/Mochi_preview_comfy' to 'models/diffusion_models/mochi'", },
                ),
                "vae": (
                    [   
                        "mochi_preview_vae_bf16.safetensors",
                    ],
                    {"tooltip": "Downloads from 'https://huggingface.co/Kijai/Mochi_preview_comfy' to 'models/vae/mochi'", },
                ),
                 "precision": (["fp8_e4m3fn","fp8_e4m3fn_fast","fp16", "fp32", "bf16"],
                    {"default": "fp8_e4m3fn", }),
                "attention_mode": (["sdpa","flash_attn","sage_attn", "comfy"],
                ),
            },
            "optional": {
                "trigger": ("CONDITIONING", {"tooltip": "Dummy input for forcing execution order",}),
            },
        }

    RETURN_TYPES = ("MOCHIMODEL", "MOCHIVAE",)
    RETURN_NAMES = ("mochi_model", "mochi_vae" )
    FUNCTION = "loadmodel"
    CATEGORY = "MochiWrapper"
    DESCRIPTION = "Downloads and loads the selected Mochi model from Huggingface"

    def loadmodel(self, model, vae, precision, attention_mode, trigger=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Transformer model
        model_download_path = os.path.join(folder_paths.models_dir, 'diffusion_models', 'mochi')
        model_path = os.path.join(model_download_path, model)
   
        repo_id = "kijai/Mochi_preview_comfy"
        
        if not os.path.exists(model_path):
            log.info(f"Downloading mochi model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{model}*"],
                local_dir=model_download_path,
                local_dir_use_symlinks=False,
            )
        # VAE
        vae_download_path = os.path.join(folder_paths.models_dir, 'vae', 'mochi')
        vae_path = os.path.join(vae_download_path, vae)

        if not os.path.exists(vae_path):
            log.info(f"Downloading mochi VAE to: {vae_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{vae}*"],
                local_dir=vae_download_path,
                local_dir_use_symlinks=False,
            )

        model = T2VSynthMochiModel(
            device=device,
            offload_device=offload_device,
            vae_stats_path=os.path.join(script_directory, "configs", "vae_stats.json"),
            dit_checkpoint_path=model_path,
            weight_dtype=dtype,
            fp8_fastmode = True if precision == "fp8_e4m3fn_fast" else False,
            attention_mode=attention_mode
        )
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            vae = Decoder(
                    out_channels=3,
                    base_channels=128,
                    channel_multipliers=[1, 2, 4, 6],
                    temporal_expansions=[1, 2, 3],
                    spatial_expansions=[2, 2, 2],
                    num_res_blocks=[3, 3, 4, 6, 3],
                    latent_dim=12,
                    has_attention=[False, False, False, False, False],
                    padding_mode="replicate",
                    output_norm=False,
                    nonlinearity="silu",
                    output_nonlinearity="silu",
                    causal=True,
                )
        vae_sd = load_torch_file(vae_path)
        if is_accelerate_available:
            for key in vae_sd:
                set_module_tensor_to_device(vae, key, dtype=torch.float32, device=device, value=vae_sd[key])
        else:
            vae.load_state_dict(vae_sd, strict=True)
            vae.eval().to(torch.bfloat16).to("cpu")
        del vae_sd

        return (model, vae,)
    
class MochiModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The name of the checkpoint (model) to load.",}),
                "precision": (["fp8_e4m3fn","fp8_e4m3fn_fast","fp16", "fp32", "bf16"], {"default": "fp8_e4m3fn"}),
                "attention_mode": (["sdpa","flash_attn","sage_attn", "comfy"],),
            },
            "optional": {
                "trigger": ("CONDITIONING", {"tooltip": "Dummy input for forcing execution order",}),
            },
        }
    RETURN_TYPES = ("MOCHIMODEL",)
    RETURN_NAMES = ("mochi_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "MochiWrapper"

    def loadmodel(self, model_name, precision, attention_mode, trigger=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        model = T2VSynthMochiModel(
            device=device,
            offload_device=offload_device,
            vae_stats_path=os.path.join(script_directory, "configs", "vae_stats.json"),
            dit_checkpoint_path=model_path,
            weight_dtype=dtype,
            fp8_fastmode = True if precision == "fp8_e4m3fn_fast" else False,
            attention_mode=attention_mode
        )

        return (model, )

class MochiVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "The name of the checkpoint (vae) to load."}),
            },
        }

    RETURN_TYPES = ("MOCHIVAE",)
    RETURN_NAMES = ("mochi_vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "MochiWrapper"

    def loadmodel(self, model_name):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        vae_path = folder_paths.get_full_path_or_raise("vae", model_name)

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            vae = Decoder(
                    out_channels=3,
                    base_channels=128,
                    channel_multipliers=[1, 2, 4, 6],
                    temporal_expansions=[1, 2, 3],
                    spatial_expansions=[2, 2, 2],
                    num_res_blocks=[3, 3, 4, 6, 3],
                    latent_dim=12,
                    has_attention=[False, False, False, False, False],
                    padding_mode="replicate",
                    output_norm=False,
                    nonlinearity="silu",
                    output_nonlinearity="silu",
                    causal=True,
                )
        vae_sd = load_torch_file(vae_path)
        if is_accelerate_available:
            for key in vae_sd:
                set_module_tensor_to_device(vae, key, dtype=torch.float32, device=device, value=vae_sd[key])
        else:
            vae.load_state_dict(vae_sd, strict=True)
            vae.eval().to(torch.bfloat16).to("cpu")
        del vae_sd

        return (vae,)
    
class MochiTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CLIP",)
    RETURN_NAMES = ("conditioning", "clip", )
    FUNCTION = "process"
    CATEGORY = "MochiWrapper"

    def process(self, clip, prompt, strength=1.0, force_offload=True):
        max_tokens = 256
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        clip.tokenizer.t5xxl.pad_to_max_length = True
        clip.tokenizer.t5xxl.max_length = max_tokens
        clip.cond_stage_model.t5xxl.return_attention_masks = True
        clip.cond_stage_model.t5xxl.enable_attention_masks = True
        clip.cond_stage_model.t5_attention_mask = True
        clip.cond_stage_model.to(load_device)
        tokens = clip.tokenizer.t5xxl.tokenize_with_weights(prompt, return_word_ids=True)
        
        try:
            embeds, _, attention_mask = clip.cond_stage_model.t5xxl.encode_token_weights(tokens)
        except:
            NotImplementedError("Failed to get attention mask from T5, is your ComfyUI up to date?")

        if embeds.shape[1] > 256:
            raise ValueError(f"Prompt is too long, max tokens supported is {max_tokens} or less, got {embeds.shape[1]}")
        embeds *= strength
        if force_offload:
            clip.cond_stage_model.to(offload_device)

        t5_embeds = {
            "embeds": embeds,
            "attention_mask": attention_mask["attention_mask"].bool(),
        }
        return (t5_embeds, clip,)
    

class MochiSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MOCHIMODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "width": ("INT", {"default": 848, "min": 128, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 49, "min": 7, "max": 1024, "step": 6}),
                "steps": ("INT", {"default": 50, "min": 2}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                #"batch_cfg": ("BOOLEAN", {"default": False, "tooltip": "Enable batched cfg"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("model", "samples",)
    FUNCTION = "process"
    CATEGORY = "MochiWrapper"

    def process(self, model, positive, negative, steps, cfg, seed, height, width, num_frames):
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        args = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "mochi_args": {
                "sigma_schedule": linear_quadratic_schedule(steps, 0.025),
                "cfg_schedule": [cfg] * steps,
                "num_inference_steps": steps,
                "batch_cfg": False,
            },
            "positive_embeds": positive,
            "negative_embeds": negative,
            "seed": seed,
        }
        latents = model.run(args)
    
        mm.soft_empty_cache()

        return ({"samples": latents},)
    
class MochiDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("MOCHIVAE",),
            "samples": ("LATENT", ),
            "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
            "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Auto size based on height and width, default is half the size"}),
            "frame_batch_size": ("INT", {"default": 6, "min": 1, "max": 64, "step": 1, "tooltip": "Number of frames in latent space (downscale factor is 6) to decode at once"}),
            "tile_sample_min_height": ("INT", {"default": 240, "min": 16, "max": 2048, "step": 8, "tooltip": "Minimum tile height, default is half the height"}),
            "tile_sample_min_width": ("INT", {"default": 424, "min": 16, "max": 2048, "step": 8, "tooltip": "Minimum tile width, default is half the width"}),
            "tile_overlap_factor_height": ("FLOAT", {"default": 0.1666, "min": 0.0, "max": 1.0, "step": 0.001}),
            "tile_overlap_factor_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MochiWrapper"

    def decode(self, vae, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, 
               tile_overlap_factor_width, auto_tile_size, frame_batch_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        intermediate_device = mm.intermediate_device()
        samples = samples["samples"]
        samples = samples.to(torch.bfloat16).to(device)

        B, C, T, H, W = samples.shape

        self.tile_overlap_factor_height = tile_overlap_factor_height if not auto_tile_size else 1 / 6
        self.tile_overlap_factor_width = tile_overlap_factor_width if not auto_tile_size else 1 / 5

        self.tile_sample_min_height = tile_sample_min_height if not auto_tile_size else H // 2 * 8
        self.tile_sample_min_width = tile_sample_min_width if not auto_tile_size else W // 2 * 8

        self.tile_latent_min_height = int(self.tile_sample_min_height / 8)
        self.tile_latent_min_width = int(self.tile_sample_min_width / 8)

        
        def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            blend_extent = min(a.shape[3], b.shape[3], blend_extent)
            for y in range(blend_extent):
                b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                    y / blend_extent
                )
            return b

        def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            blend_extent = min(a.shape[4], b.shape[4], blend_extent)
            for x in range(blend_extent):
                b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                    x / blend_extent
                )
            return b
        
        def decode_tiled(samples):
            overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
            overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
            blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
            blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
            row_limit_height = self.tile_sample_min_height - blend_extent_height
            row_limit_width = self.tile_sample_min_width - blend_extent_width

            # Split z into overlapping tiles and decode them separately.
            # The tiles have an overlap to avoid seams between tiles.
            comfy_pbar = ProgressBar(len(range(0, H, overlap_height)))
            rows = []
            for i in tqdm(range(0, H, overlap_height), desc="Processing rows"):
                row = []
                for j in tqdm(range(0, W, overlap_width), desc="Processing columns", leave=False):
                    time = []
                    for k in tqdm(range(T // frame_batch_size), desc="Processing frames", leave=False):
                        remaining_frames = T % frame_batch_size
                        start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                        end_frame = frame_batch_size * (k + 1) + remaining_frames
                        tile = samples[
                            :,
                            :,
                            start_frame:end_frame,
                            i : i + self.tile_latent_min_height,
                            j : j + self.tile_latent_min_width,
                        ]
                        tile = vae(tile)
                        time.append(tile)
                    row.append(torch.cat(time, dim=2))
                rows.append(row)
                comfy_pbar.update(1)

            result_rows = []
            for i, row in enumerate(tqdm(rows, desc="Blending rows")):
                result_row = []
                for j, tile in enumerate(tqdm(row, desc="Blending tiles", leave=False)):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = blend_v(rows[i - 1][j], tile, blend_extent_height)
                    if j > 0:
                        tile = blend_h(row[j - 1], tile, blend_extent_width)
                    result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
                result_rows.append(torch.cat(result_row, dim=4))

            return torch.cat(result_rows, dim=3)
        
        vae.to(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=torch.bfloat16):
            if enable_vae_tiling and frame_batch_size > T:
                logging.warning(f"Frame batch size is larger than the number of samples, setting to {T}")
                frame_batch_size = T
                frames = decode_tiled(samples)
            elif not enable_vae_tiling:
                logging.warning("Attempting to decode without tiling, very memory intensive")
                frames = vae(samples)
            else:
                logging.info("Decoding with tiling")
                frames = decode_tiled(samples)
                
        vae.to(offload_device)

        frames = frames.float()
        frames = (frames + 1.0) / 2.0
        frames.clamp_(0.0, 1.0)

        frames = rearrange(frames, "b c t h w -> (t b) h w c").to(intermediate_device)

        return (frames,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMochiModel": DownloadAndLoadMochiModel,
    "MochiSampler": MochiSampler,
    "MochiDecode": MochiDecode,
    "MochiTextEncode": MochiTextEncode,
    "MochiModelLoader": MochiModelLoader,
    "MochiVAELoader": MochiVAELoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMochiModel": "(Down)load Mochi Model",
    "MochiSampler": "Mochi Sampler",
    "MochiDecode": "Mochi Decode",
    "MochiTextEncode": "Mochi TextEncode",
    "MochiModelLoader": "Mochi Model Loader",
    "MochiVAELoader": "Mochi VAE Loader",
    }
