import os
import threading
from typing import AsyncGenerator
from namo.api.base import VLBase
from loguru import logger
import torch
from termcolor import colored
from transformers import TextStreamer
from transformers import AutoProcessor
from namo.models.namo import NamoForCausalLM
from namo.models.configuration_namo import NamoConfig
from namo.utils.infer_utils import CallbackStreamer, load_multi_images_maybe
from namo.utils.process_utils import convert_image_tags, tokenizer_image_token
from loguru import logger
from huggingface_hub import hf_hub_download, snapshot_download
from namo.utils.process_utils import smart_resize_v1
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from namo.utils.infer_utils import url_to_image
from transformers import TextIteratorStreamer


class NamoVL(VLBase):
    def __init__(
        self,
        model_path=None,
        processor_path=None,
        device="auto",
        system_msg="You are Namo small VLM model, trained by NAMO. You can look images and with great OCR ability.",
    ):
        super().__init__(model_path, processor_path, device)
        # default: Load the model on the available device(s)
        self.default_sys = {"role": "system", "content": system_msg}
        self.history_msgs = [self.default_sys]

    def load_model(self, model_path):
        if model_path is None:
            model_path = "checkpoints/Namo-500M-V1"
        if not os.path.exists(model_path):
            logger.info(f"downloading model from huggingface into: {model_path}")
            snapshot_download(
                repo_id="lucasjin/Namo-500M-V1",
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
        model = NamoForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            # device_map="auto"
        )
        model.eval().to(self.device)
        logger.info(f"model loaded from: {model_path}")
        return model

    def load_processor(self, processor_path):
        processor = self.model.get_vision_tower().image_processor
        self.image_processor = processor
        self.tokenizer = self.model.get_namo().tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.encode(
                self.tokenizer.pad_token
            )
        return processor

    def build_chat_prompt(self, messages, tokenizer):
        converted = []
        for msg in messages:
            if msg["role"] == "system":
                converted.append(msg)
            elif msg["role"] == "assistant":
                converted.append(msg["content"])
            else:
                parts = []
                # check if content['text'] already contains image tag
                # do not convert tag
                imgs_num = 0
                txt = ""
                if isinstance(msg["content"], str):
                    txt = msg["content"]
                else:
                    for content in msg["content"]:
                        if content["type"] == "image_url":
                            parts.append("<image>")
                            imgs_num += 1
                        elif content["type"] == "text":
                            parts.append(content["text"] + "\n")
                            txt = content["text"]
                if txt.count("<image>") == imgs_num:
                    parts = txt
                else:
                    parts = "".join(parts)
                    parts = convert_image_tags(parts)
                converted.append({"role": msg["role"], "content": parts})
        return (
            tokenizer.apply_chat_template(converted, tokenize=False)
            + "<|im_start|>assistant\n"
        )

    def get_history_images(self):
        his_images = []
        for msg in self.history_msgs:
            if isinstance(msg["content"], str):
                continue
            for content in msg["content"]:
                if content["type"] == "image_url":
                    his_images.append(content["image_url"])
        return his_images

    @staticmethod
    def msg_has_img(msg):
        if isinstance(msg["content"], list):
            return any(
                [
                    c["type"] == "image_url" and c["image_url"] is not None
                    for c in msg["content"]
                ]
            )
        return False

    def remove_history_images(self):
        hist_images = []
        for msg in self.history_msgs[::-1]:
            if self.msg_has_img(msg):
                msg_new = msg.copy()
                msg_new["content"] = [
                    itm for itm in msg["content"] if itm["type"] != "image_url"
                ]
                hist_images.append(msg_new)
            else:
                hist_images.append(msg)
        self.history_msgs = hist_images[::-1]

    def get_images_history_or_none(self):
        his_images = []
        for msg in self.history_msgs:
            if isinstance(msg["content"], list):
                for itm in msg["content"]:
                    if itm["type"] == "image_url":
                        his_images.append(itm["image_url"])
        return his_images if len(his_images) > 0 else None

    def generate(
        self,
        prompt,
        images,
        stream=True,
        max_size=700,
        verbose=False,
        prevent_more_image=True,
        keep_history=True,
    ):

        if images is not None:
            crt_images = load_multi_images_maybe(images)
            if keep_history:
                if prevent_more_image:
                    # will delete previous all images.
                    self.remove_history_images()
                    images_in = crt_images
                else:
                    logger.warning(
                        "you have set prevent_more_image=False, current more can not handle history have many images, the result would be wrose."
                    )
                    images_in = crt_images + self.get_history_images()
            else:
                self.history_msgs = [self.default_sys]
                images_in = crt_images

            # print(images)
            self.image_processor.size["longest_edge"] = max_size

            pixel_values = [
                self.image_processor.preprocess(
                    img,
                    return_tensors="pt",
                )["pixel_values"]
                .to(self.model.device)
                .to(self.model.dtype)
                for img in images_in
            ]
            self.history_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": img} for img in crt_images
                    ]
                    + [
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        else:
            if keep_history:
                his_images = self.get_images_history_or_none()
                pixel_values = [
                    self.image_processor.preprocess(
                        img,
                        return_tensors="pt",
                    )["pixel_values"]
                    .to(self.model.device)
                    .to(self.model.dtype)
                    for img in his_images
                ]
            else:
                pixel_values = None
            self.history_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            )
        if verbose:
            if pixel_values is not None:
                logger.info(f"pixel_values: {[t.shape for t in pixel_values]}")

        if keep_history and len(self.history_msgs) > 6:
            # remove on first pair from history
            self.history_msgs = [
                msg for i, msg in enumerate(self.history_msgs) if i != 1 and i != 2
            ]
        if verbose and len(self.history_msgs) > 0:
            print(self.history_msgs)

        input_templated = self.build_chat_prompt(self.history_msgs, self.tokenizer)
        if verbose:
            print(input_templated)
        response = self.generate_response(
            self.model,
            self.tokenizer,
            pixel_values,
            input_templated,
            stream=stream,
        )
        if keep_history:
            self.history_msgs.append(
                {"role": "assistant", "content": response},
            )
        return response

    def generate_response(
        self, model, tokenizer, pixels, prompt, stream=True, return_generator=False
    ):
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )
        if stream and not return_generator:
            streamer = TextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        if return_generator:
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        gen_args = {
            "pixel_values": pixels,
            "input_ids": input_ids,
            "max_new_tokens": 460,
            "do_sample": False,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer if stream or return_generator else None,
        }
        if return_generator:
            thread = threading.Thread(
                target=self.model.generate,
                kwargs=gen_args,
            )
            thread.start()
            return (new_text for new_text in streamer)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output_ids = model.generate(**gen_args)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def chat_with_request(
        self, messages, stream=True, prevent_more_image=True, verbose=False
    ):
        """
        in case we already have a messages list
        """
        messages_new = []
        images = []
        last_img_idx = 0
        for msg in messages[::-1]:
            if self.msg_has_img(msg):
                if last_img_idx >= 1 and prevent_more_image:
                    msg_new = msg.copy()
                    msg_new["content"] = [
                        itm for itm in msg["content"] if itm["type"] != "image_url"
                    ]
                    messages_new.append(msg_new)
                else:
                    for itm in msg["content"]:
                        if itm["type"] == "image_url":
                            images.append(url_to_image(itm["image_url"]["url"]))
                    messages_new.append(msg)
                last_img_idx += 1
            else:
                messages_new.append(msg)

        if prevent_more_image:
            assert (
                len(images) <= 1
            ), "if prevent more image, images at each iter should be 1."
        messages_new = messages_new[::-1]
        

        if len(images) > 0:
            pixel_values = [
                self.image_processor.preprocess(
                    img,
                    return_tensors="pt",
                )["pixel_values"]
                .to(self.model.device)
                .to(self.model.dtype)
                for img in images
            ]
        else:
            pixel_values = None

        input_templated = self.build_chat_prompt(messages_new, self.tokenizer)

        if pixel_values is not None:
            print(input_templated)
            print(images)

        if stream:
            generator = self.generate_response(
                self.model,
                self.tokenizer,
                pixel_values,
                input_templated,
                return_generator=True,
            )
            return generator
        else:
            response = self.generate_response(
                self.model, self.tokenizer, pixel_values, input_templated, stream=False
            )
            return response

    def stream_chat_with_request(self, messages):
        for chunk in self.chat_with_request(messages, stream=True):
            yield chunk
