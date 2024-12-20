"""
This file is used to extract feature of the coco val set (to test zero-shot FID).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import os
import numpy as np
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm

import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val')
    args = parser.parse_args()
    print(args)

    if args.split == "val":
        datas = MSCOCODatabase(root='/data/qihao/dataset/coco2014/val2014',
                             annFile='/data/qihao/dataset/coco2014/annotations/captions_val2014.json',
                             size=resolution)
        save_dir = f'val'
    else:
        raise NotImplementedError

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    autoencoder = libs.autoencoder.get_model('../assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)

    llm = 'clip'

    if llm=='clip':
        clip = FrozenCLIPEmbedder()
        clip.eval()
        clip.to(device)
    elif llm=='t5':
        t5 = T5Embedder(device=device)
    else:
        raise NotImplementedError

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            x, captions = data

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            if llm=='clip':
                latent, latent_and_others = clip.encode(captions)
                token_embedding = latent_and_others['token_embedding']
                token_mask = latent_and_others['token_mask']
                token = latent_and_others['tokens']
            elif llm=='t5':
                latent, latent_and_others = t5.get_text_embeddings(captions)
                token_embedding = latent_and_others['token_embedding'].to(torch.float32) * 10.0
                token_mask = latent_and_others['token_mask']
                token = latent_and_others['tokens']

            for i in range(len(captions)):
                data = {'promt': captions[i],
                        'token_embedding': token_embedding[i].detach().cpu().numpy(), 
                        'token_mask': token_mask[i].detach().cpu().numpy(),
                        'token': token[i].detach().cpu().numpy()}
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), data)


if __name__ == '__main__':
    main()
