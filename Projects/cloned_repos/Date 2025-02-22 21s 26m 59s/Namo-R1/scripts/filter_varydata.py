"""

We need get these images haded vary data.

"""

import json
import os
import sys


def filter_json_by_image(input_json_path, image_root, output_json_path):
    # 打开并读取 JSON 文件
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 存储符合条件的项目
    filtered_data = []

    for item in data:
        image_path = item.get("image")
        if image_path:  # 如果有图片路径
            # 生成完整的图片路径
            full_image_path = os.path.join(image_root, image_path)
            # 检查图片文件是否存在
            if os.path.exists(full_image_path):
                filtered_data.append(item)

    # 将过滤后的数据写入新文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"samples all: {len(filtered_data)}")


# 示例用法
input_json_path = sys.argv[1]
image_root = "data/"
output_json_path = os.path.join(os.path.dirname(input_json_path), "vary_filtered.json")

filter_json_by_image(input_json_path, image_root, output_json_path)
