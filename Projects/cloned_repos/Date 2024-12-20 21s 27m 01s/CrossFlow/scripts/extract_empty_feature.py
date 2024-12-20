"""
This file is used to extract feature of the empty prompt.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import os
import numpy as np
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder


def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    llm = 'clip'

    if llm=='clip':
        clip = FrozenCLIPEmbedder()
        clip.eval()
        clip.to(device)
    elif llm=='t5':
        t5 = T5Embedder(device=device)
    else:
        raise NotImplementedError

    save_dir = f'./'

    if llm=='clip':
        latent, latent_and_others = clip.encode(prompts)
        token_embedding = latent_and_others['token_embedding']
        token_mask = latent_and_others['token_mask']
        token = latent_and_others['tokens']
    elif llm=='t5':
        latent, latent_and_others = t5.get_text_embeddings(prompts)
        token_embedding = latent_and_others['token_embedding'].to(torch.float32) * 10.0
        token_mask = latent_and_others['token_mask']
        token = latent_and_others['tokens']

    for i in range(len(prompts)):
        data = {'token_embedding': token_embedding[i].detach().cpu().numpy(), 
                'token_mask': token_mask[i].detach().cpu().numpy(),
                'token': token[i].detach().cpu().numpy(),
                'batch_caption': prompts[i]}
        np.save(os.path.join(save_dir, f'empty_context.npy'), data)



if __name__ == '__main__':
    main()
