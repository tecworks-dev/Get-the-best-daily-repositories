import copy
from dataclasses import dataclass
import json
import os
from typing import Dict, Sequence
from PIL import Image
import jsonlines
import random
import torch
from torch.utils.data import Dataset
import transformers
from namo.dataargs import DataArguments
from namo.models.symbols import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
)
from namo.utils.process_utils import (
    convert_image_tags,
    get_suitable_size_hw,
    process_video_fixed_frames,
    resize_pad_images_to_target,
)
from namo.utils.utils import rank0_print
from namo.utils import convs as conversation_lib
from namo.utils.process_template import *


def is_dynamic_size_input(keys):
    return "shortest_edge" in keys and "longest_edge" in keys


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_suitable_size(images, longest_edge=800):
    max_width = 0
    max_height = 0
    for image in images:
        if isinstance(image, list):
            image = image[0]
        width, height = image.size
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
    return min(max(max_width, max_height), longest_edge)


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if sentence["from"] != "gpt":
                sentence["value"] = sentence["value"].lstrip("\n")
                # possiably avoid <image> or <video> token exist in second or later turn
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    images_num = sentence["value"].count(DEFAULT_IMAGE_TOKEN)
                    if images_num == 1:
                        sentence["value"] = (
                            sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                        )
                        sentence["value"] = (
                            DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                        )
                        sentence["value"] = sentence["value"].strip()
                        if "mmtag" in conversation_lib.default_conversation.version:
                            sentence["value"] = sentence["value"].replace(
                                DEFAULT_IMAGE_TOKEN,
                                "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                            )
                    else:
                        # pass
                        # make <image> <image> into: Image1<image>\nImage2<image>\n
                        sentence["value"] = convert_image_tags(sentence["value"])
                        # print(f'multi images {images_num} {sentence["value"]}')
                    # multi images, keep they same as original position
                elif DEFAULT_VIDEO_TOKEN in sentence["value"]:
                    # print(f'video {sentence}')
                    # force <video> into video_frames_num * '<image> ' indicates frames
                    sentence["value"] = (
                        sentence["value"].replace(DEFAULT_VIDEO_TOKEN, "").strip()
                    )
                    sentence["value"] = (
                        "video sequence frames in order:\n"
                        + f"{DEFAULT_IMAGE_TOKEN} " * data_args.video_frames_num
                        + "\n"
                        + sentence["value"]
                    )
                    # make <image> <image> into: 1<image>\n2<image>\n
                    sentence["value"] = convert_image_tags(sentence["value"])
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                    )
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )
    return sources


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mistral":
        return preprocess_mistral(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []
        for data in data_path:
            if data.endswith("jsonl"):
                with jsonlines.open(data, mode="r") as reader:
                    raw_data = [item for item in reader]
            else:
                raw_data = json.load(open(data, "r"))

            for i in raw_data:
                if "conversations" in i.keys():
                    i["id"] = len(list_data_dict)
                    i["ds"] = os.path.basename(data).split(".")[0].split("_train")[0]
                    list_data_dict.append(i)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        random.shuffle(self.list_data_dict)
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample or "video" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        while attempt < max_attempt:
            try:
                # sample an item
                data_dict = self._sample_item(i)
                # if data_dict is not None:
                break
            except Exception as e:
                attempt += 1
                print(f"Error in loading {i}, retrying...")
                import traceback

                print(e)
                traceback.print_exc()
                i = random.randint(0, len(self.list_data_dict) - 1)
        return data_dict

    def _sample_item(self, i) -> Dict[str, torch.Tensor]:
        image = None
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            # image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            # processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_file = self.list_data_dict[i]["image"]

            ds = self.list_data_dict[i]["ds"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # todo: consider multiple images input.
            if (
                ("llava" in ds and "llavar" not in ds and "llava_recap" not in ds)
                or "sharegpt4v_instruct" in ds
                or "sharegpt4v_" in ds
                or "share-captioner" in ds
                or "gemini" in ds
                or "bunny_695k" in ds
                or "allava_laion" in ds
                or "multi_llava" in ds
                or "Cambrian7M" in ds
                or "c7s-" in ds
                or "ureader_tr" in ds
            ):
                if isinstance(image_file, list):
                    image = [
                        Image.open(os.path.join(image_folder, img_f)).convert("RGB")
                        for img_f in image_file
                    ]
                else:
                    image = Image.open(os.path.join(image_folder, image_file)).convert(
                        "RGB"
                    )
            else:
                if "llavar" in ds:
                    ds = "llavar"
                elif "bunny" in ds and "bunny_695k" not in ds:
                    ds = "bunny_pretrain_laion_2m"
                elif "qa_" in ds:
                    ds = "qa_data"
                elif "sharegpt4o" in ds:
                    ds = "sharegpt4o/images"
                elif "mathv360k_cot" in ds:
                    ds = "mathv360k_cot/images"

                if isinstance(image_file, list):
                    image = [
                        Image.open(os.path.join(image_folder, ds, img_f)).convert("RGB")
                        for img_f in image_file
                    ]
                else:
                    image = Image.open(
                        os.path.join(image_folder, ds, image_file)
                    ).convert("RGB")

                # todo: checking image validness here.
                def is_valid_image(img):
                    width, height = img.size
                    # must bigger than 28 pixels
                    if width > 14 and height > 14:
                        return True
                    else:
                        return False

                if isinstance(image, list):
                    for img in image:
                        if not is_valid_image(img):
                            rank0_print(f"Invalid image found, passing... {img.size}")
                            raise ValueError(f"Invalid image found {img.size}")
                else:
                    if not is_valid_image(image):
                        rank0_print(f"Invalid image found, passing... {image.size}")
                        raise ValueError(f"Invalid image found {image.size}")

            if self.data_args.image_aspect_ratio == "pad":
                if (
                    not (
                        processor.size and is_dynamic_size_input(processor.size.keys())
                    )
                    and not self.data_args.dynamic_size
                ):
                    # for navit we dont need pad
                    if isinstance(image, list):
                        image = [
                            expand2square(
                                i, tuple(int(x * 255) for x in processor.image_mean)
                            )
                            for i in image
                        ]
                        # only preprocess item by item when fixed sizes, otherwise do it in batch
                        image = processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ]
                    else:
                        image = expand2square(
                            image, tuple(int(x * 255) for x in processor.image_mean)
                        )
                        # only preprocess item by item when fixed sizes, otherwise do it in batch
                        image = processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]
            else:
                # print(processor.size)
                # does multiple images can be handled here?
                if (
                    not (
                        processor.size and is_dynamic_size_input(processor.size.keys())
                    )
                    and not self.data_args.dynamic_size
                ):
                    # only preprocess item by item when fixed sizes, otherwise do it in batch
                    if isinstance(image, list):
                        image = processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ]
                    else:
                        image = processor.preprocess(image, return_tensors="pt")[
                            "pixel_values"
                        ][0]

            # if muti img and not same with image token and real images num, force it.
            if isinstance(image_file, list) and sources[0]["conversations"][0][
                "value"
            ].count(DEFAULT_IMAGE_TOKEN) != len(image_file):
                a = sources[0]["conversations"][0]["value"].replace(
                    DEFAULT_IMAGE_TOKEN, ""
                )
                sources[0]["conversations"][0]["value"] = (
                    f"{DEFAULT_IMAGE_TOKEN} " * len(image_file) + a
                )
            elif (
                isinstance(image_file, str)
                and sources[0]["conversations"][0]["value"].count(DEFAULT_IMAGE_TOKEN)
                > 1
            ):
                # sometimes single image can have multiple <image> tag
                print(f"data single turn but got multiple <image>: {sources}")
                a = sources[0]["conversations"][0]["value"].replace(
                    DEFAULT_IMAGE_TOKEN, ""
                )
                sources[0]["conversations"][0]["value"] = f"{DEFAULT_IMAGE_TOKEN} " + a

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        elif "image" not in sources[0] and "video" in sources[0]:
            # print("video sample")
            video_file = self.list_data_dict[i]["video"]
            ds = self.list_data_dict[i]["ds"]
            image_folder = self.data_args.image_folder
            video_file = os.path.join(image_folder, ds, video_file)

            video = process_video_fixed_frames(
                video_file, self.data_args.video_fps, self.data_args.video_frames_num
            )
            processor = self.data_args.image_processor
            image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        # print(f'sources : {sources}')
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i] or "video" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if "image" in self.list_data_dict[i] or "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
                if self.data_args.dynamic_size:
                    data_dict["image"] = Image.new("RGB", (448, 448), (0, 0, 0))
                else:
                    data_dict["image"] = torch.zeros(
                        3, crop_size["height"], crop_size["width"]
                    )
            else:
                processor = self.data_args.image_processor
                size = processor.size
                if not (
                    processor.size and is_dynamic_size_input(processor.size.keys())
                ):
                    data_dict["image"] = torch.zeros(3, size["height"], size["width"])
                else:
                    # fake for pure text in navit
                    data_dict["image"] = Image.new("RGB", (448, 448), (0, 0, 0))
        # print(data_dict)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            if isinstance(instances[0]["image"], torch.Tensor):
                images = [instance["image"] for instance in instances]
                if all(
                    x is not None and x.shape[-2:] == images[0].shape[-2:]
                    for x in images
                ):
                    batch["pixel_values"] = torch.cat(
                        [i.unsqueeze(0) if len(i.shape) == 3 else i for i in images],
                        dim=0,
                    )
                else:
                    batch["pixel_values"] = images
            else:
                # handle various sizes image inputs
                images = [instance["image"] for instance in instances]
                images = [
                    item
                    for sublist in images
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
                if self.data_args.dynamic_size:
                    # size = get_suitable_size(images)

                    if self.data_args.native_size:
                        batch["pixel_values"] = [
                            self.data_args.image_processor.preprocess(
                                img,
                                return_tensors="pt",
                            )["pixel_values"]
                            for img in images
                        ]
                    else:
                        size = get_suitable_size_hw(
                            images, longest_edge=self.data_args.longest_edge
                        )
                        images = resize_pad_images_to_target(images, size)
                        images_tensor = self.data_args.image_processor.preprocess(
                            images,
                            return_tensors="pt",
                            # size={"width": size, "height": size},
                        )
                        batch["pixel_values"] = images_tensor["pixel_values"]
                else:
                    images_tensor = self.data_args.image_processor.preprocess(
                        images, return_tensors="pt"
                    )
                    batch["pixel_values"] = images_tensor["pixel_values"]

                if not isinstance(batch["pixel_values"], list):
                    if "pixel_attention_mask" in images_tensor:
                        batch["pixel_attention_mask"] = images_tensor[
                            "pixel_attention_mask"
                        ][0]
                # this will goes to Navit
                # print('does it got pixeltteionamsk? ------------>')
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, data_args=data_args
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
