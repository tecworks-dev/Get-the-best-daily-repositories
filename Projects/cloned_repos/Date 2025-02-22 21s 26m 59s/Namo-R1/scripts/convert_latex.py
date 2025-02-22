"""
convert

https://huggingface.co/datasets/linxy/LaTeX_OCR

"""

import json
from datasets import load_dataset
import uuid
import os

all_res = []

ds = load_dataset("parquet", data_files="latex/full/*.parquet")


def build_item(item, idx):
    # print(item)
    new_item = {
        "id": f"{idx}",
        "image": f"images/{idx}.jpg",
    }
    new_item["conversations"] = []

    os.makedirs("latex_ocr/images", exist_ok=True)
    item["image"] = item["image"].convert("RGB")

    item["image"].save(os.path.join("latex_ocr", new_item["image"]))

    new_item["conversations"].append(
        {"from": "human", "value": "<image>\nPlease convert this into latex format."}
    )
    new_item["conversations"].append({"from": "gpt", "value": item["text"]})
    return new_item


print(ds.items())
idx = 0
for item in ds["train"]:
    new_item = build_item(item, f"latex-ocr-full-train-{idx}")
    all_res.append(new_item)
    idx += 1

# idx = 0
# for item in ds['test']:
#     new_item = build_item(item, f'latex-ocr-full-test-{idx}')
#     all_res.append(new_item)
#     idx += 1

ds = load_dataset("linxy/LaTeX_OCR", "human_handwrite", streaming=True)

idx = 0
for item in ds["train"]:
    new_item = build_item(item, f"latex-ocr-human_handwrite-train-{idx}")
    all_res.append(new_item)
    idx += 1

# idx = 0
# for item in ds['test']:
#     new_item = build_item(item, f'latex-ocr-human_handwrite-test-{idx}')
#     all_res.append(new_item)
#     idx += 1

print(f"all res: {len(all_res)}")
file_path = f"latex_ocr.json"
with open(file_path, "w") as f:
    json.dump(all_res, f, ensure_ascii=False, indent=2)
