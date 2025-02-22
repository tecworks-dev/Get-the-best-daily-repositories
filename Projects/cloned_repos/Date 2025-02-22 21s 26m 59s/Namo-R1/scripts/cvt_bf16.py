import torch
from transformers import AutoModel, AutoTokenizer
import argparse
import os


def convert_and_save_bf16(model_path, output_dir=None):

    try:
        if output_dir is None:
            output_dir = model_path.strip("/") + "_bf16"
        os.makedirs(output_dir, exist_ok=True)
        print(f"⏳ 正在加载原始模型来自: {model_path}")

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # 初始加载为BF16
            # device_map="auto",  # 自动分配设备
            low_cpu_mem_usage=True,  # 优化内存使用
            trust_remote_code=True,
        )

        print("🔧 正在转换模型权重到BF16...")
        model = model.to(torch.bfloat16)

        print(f"💾 正在保存BF16模型到: {output_dir}")
        model.save_pretrained(
            output_dir,
            safe_serialization=True,  # 使用safetensors格式
            max_shard_size="6GB",  # 分片大小
        )

        # 保存tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print("passing save tokenzier.")

        print("✅ 转换完成！保存内容：")
        print(f"   - 模型权重: {output_dir}/pytorch_model*.bin")
        print(f"   - 配置文件: {output_dir}/config.json")
        print(f"   - Tokenizer文件: {output_dir}/tokenizer.*")

    except Exception as e:
        print(f"❌ 错误发生: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换HF模型到BF16格式")
    parser.add_argument(
        "model_path",
        type=str,
        help="输入模型路径（本地目录或HF Hub名称）",
    )
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    convert_and_save_bf16(args.model_path, args.output_dir)
