import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch
import numpy as np
import random
import shutil
import tempfile
from PIL import Image
from huggingface_hub import snapshot_download
import folder_paths

from .OmniGen import OmniGenPipeline

model_path = os.path.join(folder_paths.models_dir, "OmniGen", "Shitao", "OmniGen-v1")

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def generate_random_name(prefix:str, suffix:str, length:int) -> str:
    name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(length))
    return prefix + name + suffix

def save_tmp_image(image:Image, temp_dir:str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=temp_dir) as f:
        image.save(f.name)
    return f.name

class OmniGen_Model:
    def __init__(self, quantization):
        self.quantization = quantization
        self.pipe = OmniGenPipeline.from_pretrained(
                    model_path,
                    Quantization=quantization
                )


class DZ_OmniGenV1:

    def __init__(self):
        self.NODE_NAME = "OmniGen Wrapper"
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        dtype_list = ["default", "int8"]
        return {
            "required": {
                "dtype": (dtype_list,),
                "prompt": ("STRING", {
                    "default": "input image as {image_1}, e.g.", "multiline":True
                }),
                "width": ("INT", {
                    "default": 1024, "min": 16, "max": 2048, "step": 16
                }),
                "height": ("INT", {
                    "default": 1024, "min": 16, "max": 2048, "step": 16
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 5.0, "step": 0.1
                }),
                "img_guidance_scale": ("FLOAT", {
                    "default": 1.6, "min": 1.0, "max": 2.0, "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 25, "min": 1, "max": 100, "step": 1
                }),
                "separate_cfg_infer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Can save memory when generating images of large size at the expense of slower inference"
                }),
                "use_kv_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable kv cache to speed up the inference"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 1e18, "step": 1
                }),
                "cache_model": ("BOOLEAN", {
                    "default": False, "tooltip": "Cache model in VRM to save loading time"
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run_omnigen"
    CATEGORY = '😺dzNodes/OmniGen Wrapper'

    def run_omnigen(self, dtype, prompt, width, height, guidance_scale, img_guidance_scale,
                    steps, separate_cfg_infer, use_kv_cache, seed, cache_model,
                    image_1=None, image_2=None, image_3=None
                 ):

        if not os.path.exists(os.path.join(model_path, "model.safetensors")):
            snapshot_download("Shitao/OmniGen-v1",local_dir=model_path)

        quantization = True if dtype == "int8" else False
        if self.model is None or self.model.quantization != quantization:
            self.model = OmniGen_Model(quantization)

        temp_dir = os.path.join(folder_paths.get_temp_directory(), generate_random_name('_ominigen_', '_temp', 16))
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        try:
            os.makedirs(temp_dir)
        except Exception as e:
            print(f"Error: {self.NODE_NAME} skipped, because {e}", message_type='error')
            return (None,)

        input_images = []
        if image_1 is not None:
            input_images.append(save_tmp_image(tensor2pil(image_1), temp_dir))
            prompt = prompt.replace("{image_1}", "<img><|image_1|></img>")
        if image_2 is not None:
            input_images.append(save_tmp_image(tensor2pil(image_2), temp_dir))
            prompt = prompt.replace("{image_2}", "<img><|image_2|></img>")
        if image_3 is not None:
            input_images.append(save_tmp_image(tensor2pil(image_2), temp_dir))
            prompt = prompt.replace("{image_3}", "<img><|image_3|></img>")
        if len(input_images) == 0:
            input_images = None

        # Generate image
        output = self.model.pipe(
            prompt=prompt,
            input_images=input_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=steps,
            separate_cfg_infer=separate_cfg_infer,  # set False can speed up the inference process
            use_kv_cache=use_kv_cache,
            seed=seed,
        )
        ret_image = np.array(output[0]) / 255.0
        ret_image = torch.from_numpy(ret_image)

        if not cache_model:
            self.model = None
            import gc
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        shutil.rmtree(temp_dir)

        return (ret_image.unsqueeze(0),)

NODE_CLASS_MAPPINGS = {
    "dzOmniGenWrapper": DZ_OmniGenV1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "dzOmniGenWrapper": "😺dz: OmniGen Wrapper"
}