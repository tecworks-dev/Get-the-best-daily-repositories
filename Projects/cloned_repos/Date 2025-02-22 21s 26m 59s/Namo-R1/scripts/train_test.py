from transformers import AutoConfig
from transformers import TextStreamer
from namo.models.namo import NamoForCausalLM
from namo.models.configuration_namo import NamoConfig
from namo.utils.infer_utils import load_multi_images_maybe
from namo.utils.process_utils import tokenizer_image_token
import torch

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

text_config = AutoConfig.from_pretrained(
    "checkpoints/Qwen2.5-0.5B-Instruct", trust_remote_code=True
)
vision_config = AutoConfig.from_pretrained(
    # "checkpoints/aimv2-large-patch14-native", trust_remote_code=True
    "checkpoints/aimv2-l-native-trained-base",
    trust_remote_code=True,
)


config = NamoConfig(text_config=text_config, vision_config=vision_config)

namo_model = NamoForCausalLM(config=config).to(device)
namo_model.namo.load_conn_ve_llm_weights(
    "checkpoints/namo-qwen2-500m-aimv2-native-conn-ve-mlp2x_gelu/checkpoint-2500/conn_ve_llm.bin"
)
image_processor = namo_model.get_vision_tower().image_processor

images = load_multi_images_maybe("images/cats.jpg")
pixel_values = (
    image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    .to(namo_model.dtype)
    .to(namo_model.device)
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
        # "hello, how are you.",
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
        max_new_tokens=100,
        streamer=streamer,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
print(f"final output:\n{outputs}")

# namo_model.generate(pixel_values=None, input_ids=input_ids, max_new_tokens=300)

model_path = "checkpoints/namo-500m-native"
namo_model.save_pretrained(model_path)
config.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
image_processor.save_pretrained(model_path)
