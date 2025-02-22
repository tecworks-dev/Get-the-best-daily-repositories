import ast
import random
import re

import numpy as np
from namo.models.symbols import IMAGE_TOKEN_INDEX
import torch
from PIL import Image
from loguru import logger

try:
    from decord import VideoReader, cpu
except ImportError as e:
    pass
try:
    from moviepy.editor import VideoFileClip
except ImportError as e:
    pass


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None,
    add_special_tokens=True,
):
    prompt_chunks = [
        tokenizer(chunk, add_special_tokens=add_special_tokens).input_ids
        for chunk in prompt.split("<image>")
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def process_video_fixed_frames(video_file, fps, num_frames):
    def sample_frames(frame_indices):
        total_frames = len(frame_indices)
        if total_frames > num_frames:
            chunk_size = total_frames // num_frames
            frame_indices = [
                random.sample(
                    frame_indices[
                        i * chunk_size : min((i + 1) * chunk_size, total_frames)
                    ],
                    1,
                )[0]
                for i in range(num_frames)
            ]
        else:
            frame_indices = np.interp(
                np.linspace(0, total_frames - 1, num_frames),
                np.arange(total_frames),
                frame_idx,
            ).astype(int)
        return frame_indices

    if video_file.endswith("webm"):
        video_webm = VideoFileClip(video_file)
        video_frames = np.array(list(video_webm.iter_frames()))
        duration, sample_fps = len(video_frames), round(video_webm.fps / fps)
        frame_idx = [i for i in range(0, duration, sample_fps)]
        frame_idx = sample_frames(frame_idx)
        video = video_frames[frame_idx]
        return video
    else:
        vr = VideoReader(video_file, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        frame_idx = sample_frames(frame_idx)
        # random sample 1 frame based on max video frames_num
        video = vr.get_batch(frame_idx).asnumpy()
        return video


def convert_image_tags(input_string):
    if input_string.count("<image>") <= 1:
        return input_string
    count = 0

    def replacer(match):
        nonlocal count
        count += 1
        return f"\nImage{count}:{match.group()}"

    return re.sub(r"<image>", replacer, input_string).strip()


def get_suitable_size_hw(images, longest_edge=800):
    hs = [img.height for img in images]
    ws = [img.width for img in images]
    ratios = [h / w for h, w in zip(hs, ws)]

    sorted_indices = sorted(range(len(ratios)), key=lambda i: abs(ratios[i] - 1))
    k = int(len(images) * 0.75)
    selected = sorted_indices[:k]

    selected_ratios = [ratios[i] for i in selected]
    target_ratio = np.median(selected_ratios)

    h_q3 = np.percentile([hs[i] for i in selected], 75)
    w_q3 = np.percentile([ws[i] for i in selected], 75)
    sum_hw = h_q3 + w_q3

    W_initial = sum_hw / (target_ratio + 1)
    H_initial = target_ratio * W_initial

    H = int(round(H_initial / 14) * 14)
    W = int(round(W_initial / 14) * 14)

    max_edge = max(H, W)
    if max_edge > longest_edge:
        new_max = (longest_edge // 14) * 14
        if H > W:
            H = new_max
            W = int(round(H / target_ratio / 14) * 14)
        else:
            W = new_max
            H = int(round(W * target_ratio / 14) * 14)

    H, W = max(392, H), max(392, W)
    return (H, W)


def resize_pad_images_to_target(images, target_size_hw):
    H_target, W_target = target_size_hw
    processed_images = []
    for img in images:
        W, H = img.width, img.height
        aspect_ratio = W / H
        target_aspect = W_target / H_target

        if aspect_ratio >= target_aspect:
            # 缩放宽度到目标宽度，调整高度
            new_w = W_target
            new_h = int(round(H * (new_w / W)))
        else:
            # 缩放高度到目标高度，调整宽度
            new_h = H_target
            new_w = int(round(W * (new_h / H)))

        # 调整图像尺寸
        if new_w > 0 and new_h > 0:
            resized_img = img.resize((new_w, new_h), Image.BILINEAR)

            # 创建填充后的图像
            padded_img = Image.new(img.mode, (W_target, H_target), color=0)
            padded_img.paste(resized_img, (0, 0))
        else:
            logger.info(
                f"unexpected new_h: {new_h} new_w: {new_w} got. forcely resize into target {H_target}x{W_target}."
            )
            padded_img = img.resize((W_target, H_target), Image.BILINEAR)

        processed_images.append(padded_img)

    return processed_images


def smart_resize_v1(images, longest_edge=800):
    if len(images) == 1:
        return images
    target_hw = get_suitable_size_hw(images, longest_edge)
    processed_images = resize_pad_images_to_target(images, target_hw)
    return processed_images
