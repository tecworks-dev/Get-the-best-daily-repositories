from copy import deepcopy
import io
import json
import random
import numpy as np
import torch
from PIL import Image
from namo.dataset import expand2square
from namo.models.symbols import IGNORE_INDEX

from torch.utils.data import ConcatDataset, WeightedRandomSampler
from loguru import logger

import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(
            weights=self.weights, num_samples=self.total_size, replacement=True
        )

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def dpo_concat_pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    for prefix in ["chosen_", "rejected_"]:
        batch_lens = [feat[f"{prefix}input_ids"].shape[0] for feat in features]
        max_item_length = max(batch_lens)
        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[: feat[f"{prefix}input_ids"].shape[0]] = feat[
                f"{prefix}input_ids"
            ]
            feat[f"{prefix}input_ids"] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[: feat[f"{prefix}labels"].shape[0]] = feat[f"{prefix}labels"]
            feat[f"{prefix}labels"] = temp_labels
            feat[f"{prefix}attention_mask"] = feat[f"{prefix}input_ids"].ne(pad_id)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if (
            k not in ("pixel_values", "image_flags")
            and v is not None
            and not isinstance(v, str)
        ):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ("pixel_values", "image_flags"):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(
                output
            ).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


qualities = list(range(75, 101))
jpeg_degrade_functions = {
    quality: simulate_jpeg_degradation(quality) for quality in qualities
}


def build_transform(is_train, input_size, pad2square=False, normalize_type="imagenet"):
    if normalize_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.RandomChoice(
                    [T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]
                ),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Lambda(
                        lambda img: expand2square(
                            img, tuple(int(x * 255) for x in MEAN)
                        )
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )

    return transform


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method="rand",  # for video data
        repeat_time=1,
        normalize_type="imagenet",
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f"[Dataset] num_image_token: {num_image_token}")
        logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
        logger.info(
            f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
        )

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        logger.info("Formatting inputs...Skip in lazy mode")
        assert meta["annotation"].endswith(
            "jsonl"
        ), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta["annotation"], "r") as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = random.sample(
                    self.raw_data, k=int(len(self.raw_data) * repeat_time)
                )
            if repeat_time > 1:
                repeat_time = int(repeat_time)
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        self.root = meta["root"]
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = (
                {}
            )  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if "length" in data_item:
                    token_length = data_item[
                        "length"
                    ]  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = "\n".join(
                        [temp["value"] for temp in data_item["conversations"]]
                    )
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = (
                            token_length
                            + num_image_token * (max_dynamic_patch + use_thumbnail)
                        )
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template_name == "internlm2-chat":
            preprocess_function = preprocess_internlm
        elif self.template_name == "phi3-chat":
            preprocess_function = preprocess_phi3
        elif self.template_name == "internvl2_5":
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and "s3://" in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert("RGB")

    def get_image_path(self, image_path):
        if image_path.startswith("s3://"):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    @staticmethod
    def get_longest_common_prefix_index(tensor1, tensor2):
        min_len = min(len(tensor1), len(tensor2))

        for i in range(min_len):
            if tensor1[i] != tensor2[i]:
                return i

        return min_len

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["question"]:
            data_item["question"] = "<image>\n" + data_item["question"]

        # Merge the image path
        image_path = self.get_image_path(data_item["image"])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if (
            self.dynamic_image_size
        ):  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert (
                num_patches == 1
            ), f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        chosen_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["chosen"]},
        ]
        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=True,
            ds_name=self.ds_name,
        )

        rejected_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["rejected"]},
        ]
        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=True,
            ds_name=self.ds_name,
        )

        # Create the final return dictionary
        ret = dict(
            chosen_input_ids=chosen_ret["input_ids"][0],
            chosen_labels=chosen_ret["labels"][0],
            chosen_attention_mask=chosen_ret["attention_mask"][0],
            rejected_input_ids=rejected_ret["input_ids"][0],
            rejected_labels=rejected_ret["labels"][0],
            rejected_attention_mask=rejected_ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item["image"])
        for image_path in data_item["image"]:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if (
                self.dynamic_image_size
            ):  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // num_image),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        chosen_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["chosen"]},
        ]
        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        rejected_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["rejected"]},
        ]
        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        # Create the final return dictionary
        ret = dict(
            chosen_input_ids=chosen_ret["input_ids"][0],
            chosen_labels=chosen_ret["labels"][0],
            chosen_attention_mask=chosen_ret["attention_mask"][0],
            rejected_input_ids=rejected_ret["input_ids"][0],
            rejected_labels=rejected_ret["labels"][0],
            rejected_attention_mask=rejected_ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if "<video>" not in data_item["question"]:
            data_item["question"] = "<video>\n" + data_item["question"]

        # Get the video file path
        video_file = data_item["video"]
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type="video",
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get("clip", None),
        )

        # Generate special tokens for each video frame
        special_tokens = "\n".join(
            ["Frame{}: <image>".format(i + 1) for i in range(len(image_list))]
        )
        data_item["question"] = data_item["question"].replace(
            "<video>\n", special_tokens
        )

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches

        chosen_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["chosen"]},
        ]
        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=True,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_patches,
        )

        rejected_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["rejected"]},
        ]
        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=True,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_patches,
        )

        ret = dict(
            chosen_input_ids=chosen_ret["input_ids"][0],
            chosen_labels=chosen_ret["labels"][0],
            chosen_attention_mask=chosen_ret["attention_mask"][0],
            rejected_input_ids=rejected_ret["input_ids"][0],
            rejected_labels=rejected_ret["labels"][0],
            rejected_attention_mask=rejected_ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new("RGB", (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=1,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert (
            num_patches == 1
        ), f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        chosen_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["chosen"]},
        ]
        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=True,
            ds_name=self.ds_name,
        )

        rejected_conversations = [
            {"from": "human", "value": data_item["question"]},
            {"from": "gpt", "value": data_item["rejected"]},
        ]
        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=True,
            ds_name=self.ds_name,
        )

        # Create the final return dictionary
        ret = dict(
            chosen_input_ids=chosen_ret["input_ids"][0],
            chosen_labels=chosen_ret["labels"][0],
            chosen_attention_mask=chosen_ret["attention_mask"][0],
            rejected_input_ids=rejected_ret["input_ids"][0],
            rejected_labels=rejected_ret["labels"][0],
            rejected_attention_mask=rejected_ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                if "image" in data_item and len(data_item["image"]) != 0:
                    if type(data_item["image"]) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif (
                    "video" in data_item
                    and data_item["video"] is not None
                    and data_item["video"] != ""
                ):
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if "image" in data_item:
                    if type(data_item["image"]) == list:
                        images = [self.root + item for item in data_item["image"]]
                        print(
                            f"Failed to load image: {images}, the dataset is: {self.ds_name}"
                        )
                    else:
                        if data_item["image"].startswith("s3://"):
                            data_path = self.root + data_item["image"]
                        else:
                            data_path = os.path.join(self.root, data_item["image"])
                        print(
                            f"Failed to load image: {data_path}, the dataset is: {self.ds_name}"
                        )
                elif "video" in data_item:
                    data_path = os.path.join(self.root, data_item["video"])
                    print(
                        f"Failed to load video: {data_path}, the dataset is: {self.ds_name}"
                    )
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type="imagenet",
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]["repeat_time"]
        if "max_dynamic_patch" in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]["max_dynamic_patch"]
            logger.info(
                f"max_dynamic_patch is set to {max_num} according to the meta file"
            )
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name].get("data_augment", False),
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
        )
        logger.info(f"Add dataset: {ds_name} with length: {len(dataset)}")
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset
