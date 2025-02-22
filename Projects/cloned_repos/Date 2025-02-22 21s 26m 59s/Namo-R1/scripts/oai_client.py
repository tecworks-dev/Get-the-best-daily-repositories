"""
test a simple multimodal request

if you have run a Namo model in your local:

namo server --model namo
"""

from openai import OpenAI
import base64
import os

# 配置本地API信息
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # 根据实际API路径调整
    api_key="sk-r4536ybrtb",  # 如果不需要认证可以留空
)


def encode_image(image_path):
    """将本地图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


img_f = "images/cats.jpg"
# 多模态请求（文本 + 图片）
response = client.chat.completions.create(
    model="gpt-4-vision-preview",  # 根据实际部署模型调整
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片的内容"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(img_f)}"
                    },
                },
            ],
        }
    ],
    max_tokens=300,
    stream=True,
)

# for rsp in response.choices:
#     print("AI回复：", rsp.message.content)
# print("AI回复：", response.choices[0].message.content)
print(response)
for chunk in response:
    # print(chunk)
    print(chunk.choices[0].delta.content, end="", flush=True)
