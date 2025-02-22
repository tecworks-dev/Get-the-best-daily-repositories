from transformers.models.qwen2_5_vl import Qwen2_5_VLImageProcessor
from PIL import Image
from namo.processor.image_processing_namo import NamoImageProcessor


def load_images():
    imgs = [
        "images/3001910.jpg",
        "images/3001927.jpg",
        "images/3001980.jpg",
        "images/grey.jpg",
    ]
    res = []
    for im in imgs:
        res.append(Image.open(im))
    return res


imgs = load_images()
# processor = Qwen2_5_VLImageProcessor.from_pretrained(
#     "checkpoints/Qwen2.5-VL-3B-Instruct"
# )
processor = NamoImageProcessor.from_pretrained("checkpoints/Namo-500M-V1")
inputs = processor.preprocess(images=imgs)
print(inputs)
print([i.shape for i in inputs["pixel_values"]])
