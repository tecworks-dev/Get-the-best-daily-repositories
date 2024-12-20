"""
This file is used to extract feature for visulization during training
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import os
import numpy as np
from tqdm import tqdm

import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder


def main():
    prompts = [
        'A road with traffic lights, street lights and cars.',
        'A bus driving in a city area with traffic signs.',
        'A bus pulls over to the curb close to an intersection.',
        'A group of people are walking and one is holding an umbrella.',
        'A baseball player taking a swing at an incoming ball.',
        'A dog next to a white cat with black-tipped ears.',
        'A tiger standing on a rooftop while singing and jamming on an electric guitar under a spotlight. anime illustration.',
        'A bird wearing headphones and speaking into a high-end microphone in a recording studio.',
        'A bus made of cardboard.',
        'A tower in the mountains.',
        'Two cups of coffee, one with latte art of a cat. The other has latter art of a bird.',
        'Oil painting of a robot made of sushi, holding chopsticks.',
        'Portrait of a dog wearing a hat and holding a flag that has a yin-yang symbol on it.',
        'A teddy bear wearing a motorcycle helmet and cape is standing in front of Loch Awe with Kilchurn Castle behind him. dslr photo.',
        'A man standing on the moon',
    ]
    save_dir = f'run_vis'
    os.makedirs(save_dir, exist_ok=True)

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
        data = {'promt': prompts[i],
                'token_embedding': token_embedding[i].detach().cpu().numpy(), 
                'token_mask': token_mask[i].detach().cpu().numpy(),
                'token': token[i].detach().cpu().numpy()}
        np.save(os.path.join(save_dir, f'{i}.npy'), data)


if __name__ == '__main__':
    main()
