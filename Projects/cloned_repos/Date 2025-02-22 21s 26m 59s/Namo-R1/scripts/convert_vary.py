"""
sampling vary data
"""

import json
import os
import sys
import shutil
from PIL import Image

a = sys.argv[1]


def cn():
    img_root = os.path.join(os.path.dirname(a), "pdf_cn_30w")
    sample_img_root = os.path.join(os.path.dirname(a), "pdf_cn_30w_samples")
    os.makedirs(sample_img_root, exist_ok=True)
    res = json.load(open(a, "r"))

    samples = []

    for i, itm in enumerate(res):
        img_f = os.path.join(img_root, itm["image"])
        if not os.path.exists(img_f):
            print(f"{img_f} not found")

        if i < 100:
            target_img_f = os.path.join(sample_img_root, itm["image"])
            os.makedirs(os.path.dirname(target_img_f), exist_ok=True)
            shutil.copy(img_f, target_img_f)
            samples.append(itm)
    print(f"done {len(res)}")
    file_path = a.replace(".json", "_samples.json")
    with open(file_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def en():
    img_root = os.path.join(os.path.dirname(a), "pdf_en_30w")
    sample_img_root = os.path.join(os.path.dirname(a), "pdf_en_30w_samples")
    os.makedirs(sample_img_root, exist_ok=True)
    res = json.load(open(a, "r"))

    samples = []

    new_res = []

    for i, itm in enumerate(res):
        img_f = os.path.join(img_root, itm["image"])
        if not os.path.exists(img_f):
            print(f"{img_f} not found")
            continue
        else:
            new_res.append(itm)

        if i < 100:
            target_img_f = os.path.join(sample_img_root, itm["image"])
            os.makedirs(os.path.dirname(target_img_f), exist_ok=True)
            shutil.copy(img_f, target_img_f)
            samples.append(itm)
    print(f"done {len(res)}")
    file_path = a.replace(".json", "_samples.json")
    with open(file_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    file_path = a.replace(".json", "_subset.json")
    with open(file_path, "w") as f:
        json.dump(new_res, f, ensure_ascii=False, indent=2)
    print(f"done {len(new_res)}")


def cn_subset():
    """
    choose those relatively smaller size images out
    """
    img_root = os.path.join(os.path.dirname(a), "pdf_cn_30w")
    # sample_img_root = os.path.join(os.path.dirname(a), 'pdf_cn_30w_samples')
    # os.makedirs(sample_img_root, exist_ok=True)
    res = json.load(open(a, "r"))

    samples = []

    for i, itm in enumerate(res):
        img_f = os.path.join(img_root, itm["image"])
        if not os.path.exists(img_f):
            print(f"{img_f} not found")

        image = Image.open(img_f)
        if image.size[0] < 660 or image.size[1] < 660:
            samples.append(itm)
    print(f"done {len(res)}")
    print(f"done {len(samples)}")
    file_path = a.replace(".json", "_subset.json")
    with open(file_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


# en()
cn_subset()
