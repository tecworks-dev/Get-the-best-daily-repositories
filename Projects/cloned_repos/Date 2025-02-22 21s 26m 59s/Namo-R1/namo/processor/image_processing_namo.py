from typing import Dict, List, Optional, Union
import numpy as np
from transformers.image_utils import ImageInput, is_valid_image
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from transformers.utils import is_vision_available, logging
from transformers import CLIPImageProcessor
from transformers.image_transforms import resize

logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL


class NamoImageProcessor(CLIPImageProcessor):

    model_input_names = ["pixel_values"]

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Override CLIP original resize logic, we matching to longest edge if too large
        matching to shortest if too small, if within, we do nothing.
        """
        minimal_divider = 28
        config_shortest = size.get("shortest_edge", minimal_divider)
        config_longest = size.get("longest_edge", 714)

        orig_height, orig_width = image.shape[:2]
        current_shortest = min(orig_height, orig_width)
        current_longest = max(orig_height, orig_width)

        # do nothing
        if current_shortest >= config_shortest and current_longest <= config_longest:
            # we don't apply divided with 28, not necessary
            new_height = (orig_height // minimal_divider) * minimal_divider
            new_width = (orig_width // minimal_divider) * minimal_divider
            return resize(
                image,
                size=(new_height, new_width),
                resample=resample,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )

        # Determine the appropriate scaling factor.
        # If the image is too large, scale down using the longest edge.
        if current_longest > config_longest:
            scale = config_longest / current_longest
            if current_shortest * scale < config_shortest:
                # if current shortest too small after scale, we scale to shortest
                scale = config_shortest / current_shortest
        # If the image is too small, scale up using the shortest edge.
        elif current_shortest < config_shortest:
            scale = config_shortest / current_shortest
        else:
            scale = 1.0  # This branch should not be reached.

        new_height = int(round(orig_height * scale))
        new_width = int(round(orig_width * scale))

        # if longest still excceed config_longest
        if max(new_height, new_width) > config_longest:
            # this will result restortion, but should not effect detections
            if new_width > new_height:
                new_width = config_longest
            else:
                new_height = config_longest

        # ensure divided by 28 (14*2)
        new_height = (new_height // minimal_divider) * minimal_divider
        new_width = (new_width // minimal_divider) * minimal_divider

        return resize(
            image,
            size=(new_height, new_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )


__all__ = ["NamoImageProcessor"]
