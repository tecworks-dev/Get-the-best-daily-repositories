"""
convert

https://huggingface.co/datasets/amaye15/invoices-google-ocr/
https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v2

"""

import json
from datasets import load_dataset
import uuid
import os
import ast

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

    os.makedirs("invoices-ocr/images", exist_ok=True)
    if "image" in item.keys():
        image = item["image"].convert("RGB")
    else:
        image = item["pixel_values"].convert("RGB")

    img_f = os.path.join("invoices-ocr", new_item["image"])
    if not os.path.exists(img_f):
        image.save(img_f)

    if "ocr" in item.keys():
        if len(item["ocr"]) < 1:
            return None
        text = item["ocr"][0]["text"]
    else:
        a = json.loads(item["raw_data"])
        ll = ast.literal_eval(a["ocr_words"])
        # print(ll)
        text = "\n".join(ll)

    new_item["conversations"].append(
        {
            "from": "human",
            "value": "<image>\nRead all text content visible on image in order.",
        }
    )
    new_item["conversations"].append({"from": "gpt", "value": text})
    # print(new_item)
    return new_item


print(ds.items())
ds_name = os.path.basename(root_path)
idx = 0
for item in ds["train"]:
    new_item = build_item(item, f"invoices_{ds_name}_{idx}")
    if new_item is None:
        continue
    all_res.append(new_item)
    idx += 1


print(f"all res: {len(all_res)}")
file_path = f"invoices_ocr_{ds_name}.json"
with open(file_path, "w") as f:
    json.dump(all_res, f, ensure_ascii=False, indent=2)
