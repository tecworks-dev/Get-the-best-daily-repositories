import argparse
import json
import os
import random

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ds_collections = {
    'MathVista_testmini': {
        'root': 'AI4Math/MathVista',
        'max_new_tokens': 1024,
        'min_new_tokens': 1,
        'split': 'testmini'
    },
    'MathVista_test': {
        'root': 'AI4Math/MathVista',
        'max_new_tokens': 1024,
        'min_new_tokens': 1,
        'split': 'test'
    },
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def evaluate_chat_model(filename):
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(ds_collections[ds_name]['root'])[ds_collections[ds_name]['split']]

        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            image_path = 'file://' + os.path.join('/path/to/mathvista/image/', os.path.basename(data_item['image']))
            data_item['query'] = data_item['query']
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {
                            "type": "text",
                            "text": data_item['query']
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
            })

        sampling_params = SamplingParams(temperature=0.01, top_p=0.001, top_k=1, max_tokens=2048,
                                         stop_token_ids=stop_token_ids, skip_special_tokens=False,
                                         repetition_penalty=1.0)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item['decoded_image']
            data_item['response'] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            pid = data_item['pid']
            temp[pid] = data_item

        print(f'Evaluating {ds_name} ...')
        results_file = filename
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print('Results saved to {}'.format(output_path))

        cmd = f'python eval/extract_calculate.py --output_file {results_file}'
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MathVista_testmini')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--filename', type=str, default='mathvista.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.7
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model(args.filename)
