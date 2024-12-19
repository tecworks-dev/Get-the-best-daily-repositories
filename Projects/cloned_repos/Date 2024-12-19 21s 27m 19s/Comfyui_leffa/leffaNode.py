
import numpy as np
import os
import sys

from PIL import Image

from .lib.xmodel import download_hg_model 

from .leffa.transform import LeffaTransform
from .leffa.model import LeffaModel
from .leffa.inference import LeffaInference

from .lib.ximg import *

current_folder = os.path.dirname(os.path.abspath(__file__))

class CXH_Leffa_Viton_Load:
     
    def __init__(self):
        self.vt_inference = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['franciszzj/Leffa'],),
                "viton_type": (['hd',"dc"],),
            }
        }

    RETURN_TYPES = ("CXH_Leffa_Viton_Load",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "CXH/IDM"

    def gen(self, model,viton_type):
        model = download_hg_model(model)
        inpainting = os.path.join(model,"stable-diffusion-inpainting")
        if viton_type == 'hd':
            virtual_tryon = os.path.join(model,"virtual_tryon.pth")
        else:
            virtual_tryon = os.path.join(model,"virtual_tryon_dc.pth")

        vt_model = LeffaModel(
            pretrained_model_name_or_path=inpainting,
            pretrained_model=virtual_tryon,
        )
        self.vt_inference = LeffaInference(model=vt_model)

        return (self,)



class CXH_Leffa_Viton_Run:
     
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_Leffa_Viton_Load",),
                "model":("IMAGE",),
                "cloth":("IMAGE",),
                "pose":("IMAGE",),
                "mask":("MASK",),
                "steps":("INT", {"default": 20, "min": 1, "max": 100, "step": 0.01}),
                "cfg":("FLOAT", {"default": 2.5, "min": 1, "max": 50, "step": 0.01}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "CXH/IDM"

    def gen(self, pipe,model,cloth,pose,mask,steps,cfg,seed):

        src_image = tensor2pil(model)
        ref_image = tensor2pil(cloth)
        pose_image = tensor2pil(pose)
        original_size = src_image.size

        # src_image = resize_and_center(src_image, 768, 1024)
        # ref_image = resize_and_center(ref_image, 768, 1024)

        src_image = src_image.convert("RGB")
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [pose_image],
            
        }
        data = transform(data)
        output = pipe.vt_inference(data,num_inference_steps=steps,guidance_scale = cfg,seed = seed)
        gen_image = output["generated_image"][0]
        gen_image = gen_image.resize(original_size, Image.NEAREST)
        img = pil2tensor(gen_image)

        return (img,)