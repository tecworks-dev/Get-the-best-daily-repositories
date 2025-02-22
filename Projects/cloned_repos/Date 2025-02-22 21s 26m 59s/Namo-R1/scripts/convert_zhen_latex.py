"""
convert

https://huggingface.co/datasets/linxy/LaTeX_OCR

"""

import json
from datasets import load_dataset
import uuid
import os

import sys

all_res = []

root_path = sys.argv[1]

ds = load_dataset("parquet", data_files=os.path.join(root_path, "data/*.parquet"))


def build_item(item, idx):
    # print(item)
    new_item = {
        "id": f"{idx}",
        "image": f"images/{idx}.jpg",
    }
    new_item["conversations"] = []

    os.makedirs("ZhEn_latex_ocr/images", exist_ok=True)
    item["image"] = item["image"].convert("RGB")

    item["image"].save(os.path.join("ZhEn_latex_ocr", new_item["image"]))

    new_item["conversations"].append(
        {
            "from": "human",
            "value": "<image>\nPlease convert all text in image into precise latex format.",
        }
    )
    new_item["conversations"].append({"from": "gpt", "value": item["text"]})
    return new_item


print(ds.items())
idx = 0
for item in ds["train"]:
    new_item = build_item(item, f"zh-en-latex-ocr-full-train-{idx}")
    all_res.append(new_item)
    idx += 1


print(f"all res: {len(all_res)}")
file_path = f"ZhEn_latex_ocr.json"
with open(file_path, "w") as f:
    json.dump(all_res, f, ensure_ascii=False, indent=2)
