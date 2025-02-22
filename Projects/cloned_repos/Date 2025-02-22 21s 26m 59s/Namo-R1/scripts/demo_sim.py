from transformers import AutoConfig
from transformers import TextStreamer
from namo.models.namo import NamoForCausalLM
from namo.models.configuration_namo import NamoConfig
from namo.utils.infer_utils import load_multi_images_maybe
from namo.utils.process_utils import tokenizer_image_token
import torch
from loguru import logger
import sys

"""
<|im_start|>system\nYou should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n<imag
e>\nDescribe the following image.<|im_end|><|im_start|>assistant\n
"""

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

if len(sys.argv) == 1:
    model_path = "checkpoints/namo-500m"
else:
    model_path = sys.argv[1]

logger.info(f"load namo from: {model_path}")

namo_model = NamoForCausalLM.from_pretrained(model_path).to(device)
logger.success("namo model all loaded.")
image_processor = namo_model.get_vision_tower().image_processor

# images = load_multi_images_maybe("images/cats.jpg")
images = load_multi_images_maybe("images/kobe.jpg")
pixel_values = (
    image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    .to(namo_model.device)
    .to(namo_model.dtype)
)
print(f"pixel_values: {pixel_values.shape}")
tokenizer = namo_model.get_namo().tokenizer

chat = [
    {
        "role": "system",
        "content": "You should follow the instructions carefully and explain your answers in detail.",
    },
    {"role": "user", "content": "<image>\nDescribe the following image."},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False) + "<|im_start|>assistant\n"
print(prompt)

input_ids = (
    tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt",
    )
    .unsqueeze(0)
    .to(namo_model.device)
)
print(input_ids)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output_ids = namo_model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        do_sample=False,
        max_new_tokens=360,
        streamer=streamer,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
print(f"final output:\n{outputs}")
