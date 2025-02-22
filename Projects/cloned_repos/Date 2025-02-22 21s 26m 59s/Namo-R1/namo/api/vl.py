"""

A unified API interface support various VL models


"""

from namo.api.qwen2_5_vl import Qwen2_5_VL
from .namo import NamoVL


class VLInfer:
    def __init__(self, model_type="qwen2.5-vl", device="auto"):
        if "qwen2.5-vl" in model_type:
            self.model = Qwen2_5_VL()
        elif "namo" in model_type.lower():
            self.model = NamoVL(device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def generate(self, prompt, images, verbose=False):
        self.model.generate(prompt, images, verbose=verbose)
