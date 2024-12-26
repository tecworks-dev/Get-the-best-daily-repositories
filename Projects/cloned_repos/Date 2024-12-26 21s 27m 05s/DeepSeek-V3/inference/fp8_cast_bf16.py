import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant

def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    fp8_weight_names = []
    
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cuda")
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:
                scale_inv_name = f"{weight_name}_scale_inv"
                assert scale_inv_name in state_dict
                fp8_weight_names.append(weight_name)
                scale_inv = state_dict[scale_inv_name]
                new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
            else:
                new_state_dict[weight_name] = weight
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
    
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        assert scale_inv_name in weight_map
        weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    