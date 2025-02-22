import requests
from PIL import Image
import base64
from threading import Thread
import io
from transformers import TextStreamer

try:
    from datauri import DataURI
except ImportError:
    pass


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_multi_images_maybe(image_files, splitter=" "):
    if isinstance(image_files, str):
        images = image_files.split(splitter)
    else:
        images = image_files
    return [load_image(i) for i in images]


def url_to_image(img_url: str) -> Image.Image:
    if img_url.startswith("http"):
        response = requests.get(img_url)

        img_data = response.content
    elif img_url.startswith("data:"):
        img_data = DataURI(img_url).data
    else:
        img_data = base64.b64decode(img_url)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


class CallbackStreamer(TextStreamer):
    def __init__(self, tokenizer, callback=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.callback = callback

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.callback is not None:
            self.callback(text)
        super().on_finalized_text(text, stream_end)
