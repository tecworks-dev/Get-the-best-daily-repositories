"""

Extracting VE from a base trained model

before sft.

"""

from transformers import AutoConfig
from transformers import TextStreamer
from namo.models.namo import NamoForCausalLM
from namo.models.configuration_namo import NamoConfig
from namo.utils.infer_utils import load_multi_images_maybe
from namo.utils.process_utils import tokenizer_image_token
import torch
from loguru import logger
import sys

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if len(sys.argv) < 2:
        print("provide the pretrained model path please.")
        exit()

    model_path = sys.argv[1]

    logger.info(f"load namo from: {model_path}")

    namo_model = NamoForCausalLM.from_pretrained(model_path).to(device)
    logger.success("namo model all loaded.")

    ve = namo_model.get_vision_tower()
    image_processor = ve.image_processor
    tokenizer = namo_model.get_namo().tokenizer

    if "aimv2-large-patch14-native" in ve.vision_tower_name:
        save_model_path = "checkpoints/aimv2-l-native-trained-base"
    elif "aimv2-3b-p14" in model_path:
        save_model_path = "checkpoints/aimv2-3b-p14-trained-base"
    else:
        logger.info(f"unsupported vision model type: {ve.vision_tower_name}")
    ve.save_pretrained(save_model_path)
    image_processor.save_pretrained(save_model_path)
    logger.success(f"ve should be saved into: {save_model_path}")


if __name__ == "__main__":
    main()
