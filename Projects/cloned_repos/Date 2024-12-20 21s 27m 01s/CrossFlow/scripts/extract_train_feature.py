"""
This file is used to extract feature of the demo training data.
"""

import os
import shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import io
import einops
import random
import json
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder


def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main(bz = 16):

    json_path = '/path/to/JourneyDB_demo/img_text_pair.jsonl'
    root_path = '/path/to/JourneyDB_demo/imgs'

    dicts_list = []
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            dicts_list.append(json.loads(line))

    save_dir = f'train_JDBdemo'
    device = "cuda"
    recreate_folder(save_dir)

    autoencoder = libs.autoencoder.get_model('../assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)

    # CLIP model:
    clip = FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    # T5 model:
    t5 = T5Embedder(device=device)

    idx = 0
    batch_img_256 = []
    batch_img_512 = []
    batch_caption = []
    batch_name = []
    for i, sample in enumerate(tqdm(dicts_list)):
        try:
            pil_image = Image.open(os.path.join(root_path,sample['img_path']))
            caption = sample['prompt']
            img_name = sample['img_path'].replace('.jpg','')
            
            pil_image.load()
            pil_image = pil_image.convert("RGB")
        except:
            with open("failed_file.txt", 'a+') as file: 
                file.write(sample['img_path'] + "\n")
            continue

        image_256 = center_crop_arr(pil_image, image_size=256)
        image_512 = center_crop_arr(pil_image, image_size=512)

        # if True:
        #     image_id = random.randint(0,20)
        #     Image.fromarray(image_256.astype(np.uint8)).save(f"temp_img_{image_id}_256.jpg")
        #     Image.fromarray(image_512.astype(np.uint8)).save(f"temp_img_{image_id}_512.jpg")

        image_256 = (image_256 / 127.5 - 1.0).astype(np.float32)
        image_256 = einops.rearrange(image_256, 'h w c -> c h w')
        batch_img_256.append(image_256)

        image_512 = (image_512 / 127.5 - 1.0).astype(np.float32)
        image_512 = einops.rearrange(image_512, 'h w c -> c h w')
        batch_img_512.append(image_512)

        batch_caption.append(caption)
        batch_name.append(img_name)

        if len(batch_name) == bz or i == len(dicts_list) - 1:
            batch_img_256 = torch.tensor(np.stack(batch_img_256)).to(device)
            moments_256 = autoencoder(batch_img_256, fn='encode_moments').squeeze(0)
            moments_256 = moments_256.detach().cpu().numpy()

            batch_img_512 = torch.tensor(np.stack(batch_img_512)).to(device)
            moments_512 = autoencoder(batch_img_512, fn='encode_moments').squeeze(0)
            moments_512 = moments_512.detach().cpu().numpy()

            _latent_clip, latent_and_others_clip = clip.encode(batch_caption)
            token_embedding_clip = latent_and_others_clip['token_embedding'].detach().cpu().numpy()
            token_mask_clip = latent_and_others_clip['token_mask'].detach().cpu().numpy()
            token_clip = latent_and_others_clip['tokens'].detach().cpu().numpy()

            _latent_t5, latent_and_others_t5 = t5.get_text_embeddings(batch_caption)
            token_embedding_t5 = (latent_and_others_t5['token_embedding'].to(torch.float32) * 10.0).detach().cpu().numpy()
            token_mask_t5 = latent_and_others_t5['token_mask'].detach().cpu().numpy()
            token_t5 = latent_and_others_t5['tokens'].detach().cpu().numpy()

            for mt_256, mt_512, te_c, te_t, tm_c, tm_t, tk_c, tk_t, bc, bn in zip(moments_256, moments_512, token_embedding_clip, token_embedding_t5, token_mask_clip, token_mask_t5, token_clip, token_t5, batch_caption, batch_name):
                assert mt_256.shape == (8,32,32)
                assert mt_512.shape == (8,64,64)
                assert te_c.shape == (77, 768)
                assert te_t.shape == (77, 4096)
                tar_path_name = os.path.join(save_dir, f'{bn}.npy')
                if os.path.exists(tar_path_name):
                    os.remove(tar_path_name)
                data = {'image_latent_256': mt_256,
                        'image_latent_512': mt_512,
                        'token_embedding_clip': te_c, 
                        'token_embedding_t5': te_t, 
                        'token_mask_clip': tm_c,
                        'token_mask_t5': tm_t,
                        'token_clip': tk_c,
                        'token_t5': tk_t,
                        'batch_caption': bc}
                try:
                    np.save(tar_path_name, data)
                    idx += 1
                except:
                    pass
            
            batch_img_256 = []
            batch_img_512 = []
            batch_caption = []
            batch_name = []

    print(f'save {idx} files')

if __name__ == '__main__':
    main()
