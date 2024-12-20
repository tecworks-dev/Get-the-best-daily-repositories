# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
import gin
import importlib
import logging
import cv2
import matplotlib.pyplot as plt


logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader

from core.diffusion import create_diffusion
from core.models import DiT_models
from core.dataset import ImageParamsDataset
from core.utils.train_utils import load_model
from core.utils.math_utils import unnormalize_params
from scripts.prepare_data import generate

def main(cfg, generator):
    # Setup PyTorch:
    torch.manual_seed(cfg["seed"])
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = cfg["num_params"]
    model = DiT_models[cfg["model"]](input_size=latent_size).to(device)
    # load a custom DiT checkpoint from train.py:
    state_dict = load_model(cfg["ckpt_path"])
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(cfg["num_sampling_steps"]))

    # Load dataset
    dataset = ImageParamsDataset(cfg["data_root"], cfg["test_file"], cfg["params_dict_file"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False
    )
    params_dict = json.load(open(cfg["params_dict_file"]))
    idx = 0
    total_error = np.zeros(cfg["num_params"])
    for x, img_feat, img in loader:
        # sample from random noise, conditioned on image features
        img_feat = img_feat.to(device)
    
        model_kwargs = dict(y=img_feat)

        z = torch.randn(cfg["batch_size"], 1, latent_size, device=device)

        # Sample target params:
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples = samples.reshape(cfg["batch_size"], 1, -1)
        samples = samples.squeeze(1).cpu().numpy()
        x = x.squeeze(1).cpu().numpy()
        img = img.cpu().numpy()
        if cfg["run_generate"]:
            # save GT & sampled params & images
            for x_, params, img_ in zip(x, samples, img):
                # generate 3D using sampled params
                params_original = unnormalize_params(params, params_dict)
                save_dir = os.path.join(cfg["save_dir"], "{:05d}".format(idx))
                os.makedirs(save_dir, exist_ok=True)
                save_name = "sampled"
                asset, _ = generate(generator, params_original, seed=cfg["seed"], save_dir=save_dir, save_name=save_name,
                        save_blend=True, save_img=True, save_gif=False, save_mesh=True, 
                        cam_dists=cfg["r_cam_dists"], cam_elevations=cfg["r_cam_elevations"], cam_azimuths=cfg["r_cam_azimuths"], zoff=cfg["r_zoff"], 
                        resolution='256x256', sample=100)
                np.save(os.path.join(save_dir, "params.npy"), params_original)
                print("Generating model using sampled parameters. Saved in {}".format(save_dir))
                # also save GT image & GT params
                x_original = unnormalize_params(x_, params_dict)
                np.save(os.path.join(save_dir, "gt_params.npy"), x_original)
                cv2.imwrite(os.path.join(save_dir, "gt.png"), img_[:,:,::-1])
                idx += 1
        
        # calculate metrics for sampled params & GT params
        error = np.abs(x - samples)
        total_error += error
    
    # print the average error for each parameter
    avg_error = total_error / len(dataset)
    param_names = params_dict.keys()
    for param_name, error in zip(param_names, avg_error):
        print(f"{param_name}: {error:.4f}")
    # plot the error for each parameter
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 15)
    ax.barh(param_names, avg_error)
    ax.set_xlabel("Average Error")
    ax.set_ylabel("Parameters")
    ax.set_title("Average Error for Each Parameter")
    plt.yticks(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["save_dir"], "avg_error.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # load the Blender procedural generator
    OBJECTS_PATH = Path(cfg["generator_root"])
    assert OBJECTS_PATH.exists(), OBJECTS_PATH
    generator = None
    for subdir in sorted(list(OBJECTS_PATH.iterdir())):
        clsname = subdir.name.split(".")[0].strip()
        with gin.unlock_config():
            module = importlib.import_module(f"core.assets.{clsname}")
        if hasattr(module, cfg["generator"]):
            generator = getattr(module, cfg["generator"])
            logger.info("Found {} in {}".format(cfg["generator"], subdir))
            break
        logger.debug("{} not found in {}".format(cfg["generator"], subdir))
    if generator is None:
        raise ModuleNotFoundError("{} not Found.".format(cfg["generator"]))
    gen = generator(cfg["seed"])
    # create visualize dir
    os.makedirs(cfg["save_dir"], exist_ok=True)
    main(cfg, gen)
