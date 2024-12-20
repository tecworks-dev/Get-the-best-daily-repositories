# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from glob import glob
from time import time
import argparse
import os
import yaml
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from core.models import DiT_models
from core.diffusion import create_diffusion
from core.dataset import ImageParamsDataset
from core.utils.train_utils import create_logger, update_ema, requires_grad

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(cfg):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(cfg["save_dir"], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = cfg["save_dir"]
        experiment_index = len(glob(f"{save_dir}/*"))
        experiment_dir = "{}/{:03d}-{}-{}-{}".format(save_dir, experiment_index, cfg["model"], cfg["epochs"], cfg["batch_size"])  # Create an experiment folder
        checkpoint_dir = "{}/checkpoints".format(experiment_dir)  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(experiment_dir)

    # Create model:
    latent_size = cfg["num_params"]
    condition_channels = 768
    model = DiT_models[cfg["model"]](input_size=latent_size, condition_channels=condition_channels)
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=0)

    # Setup data:
    dataset = ImageParamsDataset(cfg["data_root"], cfg["train_file"], cfg["params_dict_file"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"] // accelerator.num_processes),
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info("Training for {} epochs...".format(cfg["epochs"]))
    
    # main training loop
    for epoch in range(int(cfg["epochs"])):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, img_feat, img in loader:
            # prepare the inputs
            x = x.to(device)
            img_feat = img_feat.to(device)
            x = x.unsqueeze(dim=1) # [B, 1, N]
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=img_feat)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            update_ema(ema, model)
            writer.add_scalar("train/loss", loss.item(), train_steps)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % cfg["logging_iter"] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(Step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % cfg["ckpt_iter"] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        writer.flush()
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
