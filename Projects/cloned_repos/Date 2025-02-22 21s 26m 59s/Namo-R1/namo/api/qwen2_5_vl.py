from transformers import (
    AutoTokenizer,
    AutoProcessor,
)

try:
    from qwen_vl_utils import process_vision_info
    from transformers.models.qwen2_5_vl import Qwen2_5_VLModel
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError as e:
    pass
from namo.api.base import VLBase
from loguru import logger


class Qwen2_5_VL(VLBase):
    def __init__(self, model_path=None, processor_path=None, device="auto"):
        super().__init__(model_path, processor_path, device)
        # default: Load the model on the available device(s)

    def load_model(self, model_path):
        if model_path is None:
            model_path = "checkpoints/Qwen2.5-VL-3B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        model.to(self.device)
        logger.info(f"model loaded from: {model_path}")
        return model

    def load_processor(self, processor_path):
        if processor_path is None:
            processor_path = "checkpoints/Qwen2.5-VL-3B-Instruct"
        processor = AutoProcessor.from_pretrained(processor_path)
        return processor

    def get_msg(self, text, image=None):
        if image is None:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        }

    def generate(
        self,
        prompt,
        images,
        stream=True,
        max_size=700,
        verbose=False,
        prevent_more_image=True,
    ):
        msg = self.get_msg(prompt, images)
        messages = [msg]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(output_text)
