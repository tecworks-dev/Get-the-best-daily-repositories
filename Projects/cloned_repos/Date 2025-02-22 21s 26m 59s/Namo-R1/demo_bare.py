import argparse
import json
import os
import random
import torch
from termcolor import colored
from transformers import TextStreamer
from namo.models.namo import NamoForCausalLM
from namo.models.configuration_namo import NamoConfig
from namo.utils.infer_utils import load_multi_images_maybe
from namo.utils.hf_utils import find_and_merge_lora_adapters
from namo.utils.process_utils import tokenizer_image_token
from loguru import logger


def load_model_simple(model_path):
    non_lora_bin = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_bin):
        logger.info(f"loading lora: {model_path}")
        config = NamoConfig.from_pretrained(model_path)
        model = NamoForCausalLM(config=config)
        non_lora = torch.load(non_lora_bin)
        non_lora = {k.replace("base_model.model.", ""): v for k, v in non_lora.items()}
        model.load_state_dict(non_lora, strict=False)
        model = find_and_merge_lora_adapters(model, model_path)
        return model
    else:
        return NamoForCausalLM.from_pretrained(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoints/namo-500m")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_simple(args.model_path)
    model.eval().to(device)
    image_processor = model.get_vision_tower().image_processor
    tokenizer = model.get_namo().tokenizer

    if args.eval:
        with open("images/evals.json") as f:
            for item in json.load(f):
                handle_eval_item(
                    item, model, image_processor, tokenizer, device, args.debug
                )
    else:
        run_cli(model, image_processor, tokenizer, device)


def handle_eval_item(item, model, image_processor, tokenizer, device, debug=False):
    image_path = item["image"]
    question = random.choice([item["question1"], item["question2"]])
    images = load_multi_images_maybe(image_path)
    image_processor.size["shortest_edge"] = 448
    pixel_values = (
        image_processor.preprocess(images, do_resize=True, return_tensors="pt")[
            "pixel_values"
        ]
        .to(device)
        .to(model.dtype)
    )
    if debug:
        logger.info(f"pixel_values: {pixel_values.shape}")

    chat = [
        {"role": "system", "content": "Follow instructions carefully."},
        {"role": "user", "content": f"<image>\n{question}"},
    ]
    prompt = (
        tokenizer.apply_chat_template(chat, tokenize=False) + "<|im_start|>assistant\n"
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    print(f"\nImage: {image_path}\nQ: {question}\n", end="")
    print(colored("AI: ", "yellow"), end="")
    generate_response(model, tokenizer, pixel_values, prompt)
    print("\n")


def run_cli(model, image_processor, tokenizer, device):
    DEFAULT_IMAGE = "images/cats.jpg"
    current_pixels = process_image(DEFAULT_IMAGE, image_processor, model, device)
    image_processor.size["shortest_edge"] = 448

    messages = [
        {"role": "system", "content": "Respond carefully."},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": DEFAULT_IMAGE}],
        },
    ]

    while True:
        try:
            user_input = input(colored("\nUser (txt/img_path): ", "green")).strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            if os.path.exists(user_input):
                current_pixels = process_image(
                    user_input, image_processor, model, device
                )
                messages = [
                    {"role": "system", "content": "Respond carefully."},
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": user_input}],
                    },
                ]
                print(colored("System: New image loaded", "yellow"))
                continue

            last_user_msg = next(
                (m for m in reversed(messages) if m["role"] == "user"), None
            )
            if last_user_msg and not has_text_content(last_user_msg):
                last_user_msg["content"].append({"type": "text", "text": user_input})
            else:
                messages.append(
                    {"role": "user", "content": [{"type": "text", "text": user_input}]}
                )

            prompt = build_chat_prompt(messages, tokenizer)
            print(colored("Assistant: ", "blue"), end="")
            response = generate_response(model, tokenizer, current_pixels, prompt)
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )

        except KeyboardInterrupt:
            print(colored("\nSession ended.", "red"))
            break


def process_image(path, processor, model, device):
    images = load_multi_images_maybe(path)
    return (
        processor.preprocess(images, return_tensors="pt")["pixel_values"]
        .to(device)
        .to(model.dtype)
    )


def build_chat_prompt(messages, tokenizer):
    converted = []
    for msg in messages:
        if msg["role"] == "system":
            converted.append(msg)
        else:
            parts = []
            for content in msg["content"]:
                if content["type"] == "image_url":
                    parts.append("<image>")
                elif content["type"] == "text":
                    parts.append(content["text"])
            converted.append({"role": msg["role"], "content": "\n".join(parts)})
    return (
        tokenizer.apply_chat_template(converted, tokenize=False)
        + "<|im_start|>assistant\n"
    )


def generate_response(model, tokenizer, pixels, prompt):
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(
            pixel_values=pixels,
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=360,
            streamer=streamer,
            eos_token_id=tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def has_text_content(message):
    return any(c["type"] == "text" for c in message["content"])


if __name__ == "__main__":
    main()
