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
from huggingface_hub import hf_hub_download

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from core.diffusion import create_diffusion
from core.models import DiT_models
from core.utils.train_utils import load_model
from core.utils.math_utils import unnormalize_params
from scripts.prepare_data import generate
from core.utils.dinov2 import Dinov2Model

def main(cfg, generator):
    # Setup PyTorch:
    torch.manual_seed(cfg["seed"])
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = cfg["num_params"]
    model = DiT_models[cfg["model"]](input_size=latent_size).to(device)
    # load a custom DiT checkpoint from train.py:
    # download the checkpoint if not found:
    if not os.path.exists(cfg["ckpt_path"]):
        model_dir, model_name = os.path.dirname(cfg["ckpt_path"]), os.path.basename(cfg["ckpt_path"])
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = hf_hub_download(repo_id="TencentARC/DI-PCG", 
                            local_dir=model_dir, filename=model_name)
        print("Downloading checkpoint {} from Hugging Face Hub...".format(model_name))
    print("Loading model from {}".format(cfg["ckpt_path"]))

    
    state_dict = load_model(cfg["ckpt_path"])
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(cfg["num_sampling_steps"]))
    # feature model
    feature_model = Dinov2Model()

    img_names = sorted(os.listdir(cfg["condition_img_dir"]))
    for name in img_names:
        img_path = os.path.join(cfg["condition_img_dir"], name)
        # Load condition image and extract features
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # pre-process: resize to 256x256
        img = cv2.resize(img, (256, 256))
        img = np.array(img).astype(np.uint8)

        img_feat = feature_model.encode_batch_imgs([img], global_feat=False)
        if len(img_feat.shape) == 2:
            img_feat = img_feat.unsqueeze(1)

        # Create sampling noise:
        z = torch.randn(1, 1, latent_size, device=device)
        y = img_feat

        # No classifier-free guidance:
        model_kwargs = dict(y=y)

        # Sample target params:
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples = samples[0].squeeze(0).cpu().numpy()

        # unnormalize params
        params_dict = generator.params_dict
        params_original = unnormalize_params(samples, params_dict)

        # save params
        json.dump(params_original, open("{}/{}_params.txt".format(cfg["save_dir"], name), "w"), default=str)

        # generate 3D using sampled params
        asset, _ = generate(generator, params_original, seed=cfg["seed"], save_dir=cfg["save_dir"], save_name=name,
                save_blend=True, save_img=True, save_untexture_img=True, save_gif=False, save_mesh=True, 
                cam_dists=cfg["r_cam_dists"], cam_elevations=cfg["r_cam_elevations"], cam_azimuths=cfg["r_cam_azimuths"], zoff=cfg["r_zoff"], 
                resolution='720x720', sample=200)
        print("Generating model using sampled parameters. Saved in {}".format(cfg["save_dir"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--remove_bg", type=bool, default=False)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["remove_bg"] = args.remove_bg
    
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
