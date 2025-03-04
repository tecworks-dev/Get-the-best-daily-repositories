import os
import datasets
import json
import argparse
import random
random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', help='path to the train data, hotpot_train_v1.1.json')
    parser.add_argument('--dev_data_path', help='path to the dev data, hotpot_dev_fullwiki_v1.json')
    parser.add_argument('--output_dir', help='path to the output directory')

    args = parser.parse_args()

    train_data_path = args.train_data_path
    with open(train_data_path, 'r') as f:
        lines = json.load(f)
    train_data = []
    for line in lines:
        train_data.append({
            "data_source": "hotpotqa",
            "question": line['question'],
            "ability": "qa",
            "reward_model": {
                    "style": "rule",
                    "ground_truth": line['answer']
                },
            "extra_info": {
                "id": line['_id'],
            }
        })

    dev_data_path = args.dev_data_path
    with open(dev_data_path, 'r') as f:
        lines = json.load(f)
    dev_data = []
    random.shuffle(lines)
    for line in lines[:100]:
        dev_data.append({
            "data_source": "hotpotqa",
            "question": line['question'],
            "ability": "qa",
            "reward_model": {
                "style": "rule",
                "ground_truth": line['answer']
            },
            "extra_info": {
                "id": line['_id'],
            }
        })

    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(dev_data)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_dataset.to_parquet(os.path.join(args.output_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output_dir, 'test.parquet'))