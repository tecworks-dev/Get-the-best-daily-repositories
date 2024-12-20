"""
This file contains some tools
"""
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from absl import logging
from PIL import Image, ImageDraw, ImageFont
import textwrap

def save_image_with_caption(image_tensor, caption, filename, font_size=20, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
    """
    Save an image with a caption
    """
    image_tensor = image_tensor.clone().detach()
    image_tensor = torch.clamp(image_tensor, min=0, max=1)
    image_pil = transforms.ToPILImage()(image_tensor)
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype(font_path, font_size)
    wrap_text = textwrap.wrap(caption, width=len(caption)//4 + 1)
    text_sizes = [draw.textsize(line, font=font) for line in wrap_text]
    max_text_width = max(size[0] for size in text_sizes)
    total_text_height = sum(size[1] for size in text_sizes) + 15

    new_height = image_pil.height + total_text_height + 25 
    new_image = Image.new('RGB', (image_pil.width, new_height), 'white')
    new_image.paste(image_pil, (0, 0))
    current_y = image_pil.height + 5
    draw = ImageDraw.Draw(new_image)

    for line, size in zip(wrap_text, text_sizes):
        x = (new_image.width - size[0]) / 2
        draw.text((x, current_y), line, font=font, fill='black')
        current_y += size[1] + 5
    new_image.save(filename)


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'dimr':
        from libs.model.dimr_t2i import MRModel
        return MRModel(kwargs["model_args"])
    elif name == 'dit':
        from libs.model.dit_t2i import DiT_H_2
        return DiT_H_2(kwargs["model_args"])
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def trainable_parameters(nnet):
    params_decay = []
    params_nodecay = []
    for name, param in nnet.named_parameters():
        if name.endswith(".nodecay_weight") or name.endswith(".nodecay_bias"):
            params_nodecay.append(param)
        else:
            params_decay.append(param)
    print("params_decay", len(params_decay))
    print("params_nodecay", len(params_nodecay))
    params = [
        {'params': params_decay},
        {'params': params_nodecay, 'weight_decay': 0.0}
    ]
    return params


def initialize_train_state(config, device):

    nnet = get_nnet(**config.nnet)
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(trainable_parameters(nnet), **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, return_clipScore=False, ClipSocre_model=None, config=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    clip_score_list = []

    if return_clipScore:
        assert ClipSocre_model is not None

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples, clip_score = sample_fn(mini_batch_size, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, config=config)
        samples = unpreprocess_fn(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        clip_score_list.append(accelerator.gather(clip_score)[:_batch_size])
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1
        
    if return_clipScore:
        return clip_score_list
    else:
        return None


def sample2dir_wCLIP(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, return_clipScore=False, ClipSocre_model=None, config=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    clip_score_list = []

    if return_clipScore:
        assert ClipSocre_model is not None

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples, clip_score = sample_fn(mini_batch_size, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, config=config)
        samples = unpreprocess_fn(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        clip_score_list.append(accelerator.gather(clip_score)[:_batch_size])
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1
        break
        
    if return_clipScore:
        return clip_score_list
    else:
        return None


def sample2dir_wPrompt(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, config=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    
    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples, samples_caption = sample_fn(mini_batch_size, return_caption=True, config=config)
        samples = unpreprocess_fn(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample, caption in zip(samples,samples_caption):
                try:
                    save_image_with_caption(sample, caption, os.path.join(path, f"{idx}.png"))
                except:
                    save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
