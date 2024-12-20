"""
    This file computes the clip score given image and text pair
"""
import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Normalize, Resize
import torch
import numpy as np

class ClipSocre:
    def __init__(self,device='cuda', prefix='A photo depicts', weight=1.0): # weight=2.5
        self.device = device

        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()

        self.transform = Compose([
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '
        
        self.w = weight
    
    def extract_all_images(self, images):
        images_input = self.transform(images)
        if self.device == 'cuda':
            images_input = images_input.to(torch.float16)
        image_feature = self.model.encode_image(images_input)
        return image_feature
    
    def extract_all_texts(self, texts,need_prefix):
        if need_prefix:
            c_data = clip.tokenize(self.prefix + texts, truncate=True).to(self.device)
        else:
            c_data = clip.tokenize(texts, truncate=True).to(self.device)
        text_feature = self.model.encode_text(c_data)
        return text_feature
    
    def get_clip_score(self, img, text, need_prefix=False):

        img_f = self.extract_all_images(img)
        text_f = self.extract_all_texts(text,need_prefix)
        images = img_f / torch.sqrt(torch.sum(img_f**2, axis=1, keepdims=True))
        candidates = text_f / torch.sqrt(torch.sum(text_f**2, axis=1, keepdims=True))

        clip_per = self.w * torch.clip(torch.sum(images * candidates, axis=1), 0, None)

        return clip_per
    
    def get_text_clip_score(self, text_1, text_2, need_prefix=False):
        text_1_f = self.extract_all_texts(text_1,need_prefix)
        text_2_f = self.extract_all_texts(text_2,need_prefix)

        candidates_1 = text_1_f / torch.sqrt(torch.sum(text_1_f**2, axis=1, keepdims=True))
        candidates_2 = text_2_f / torch.sqrt(torch.sum(text_2_f**2, axis=1, keepdims=True))

        per = self.w * torch.clip(torch.sum(candidates_1 * candidates_2, axis=1), 0, None)

        
        results = 'ClipS : ' + str(format(per.item(),'.4f'))

        print(results)

        return per.sum()
    
    def get_img_clip_score(self, img_1, img_2, weight = 1):

        img_f_1 = self.extract_all_images(img_1)
        img_f_2 = self.extract_all_images(img_2)

        images_1 = img_f_1 / torch.sqrt(torch.sum(img_f_1**2, axis=1, keepdims=True))
        images_2 = img_f_2 / torch.sqrt(torch.sum(img_f_2**2, axis=1, keepdims=True))

        # per = self.w * torch.clip(torch.sum(images_1 * images_2, axis=1), 0, None)
        per = weight * torch.clip(torch.sum(images_1 * images_2, axis=1), 0, None)


        return per.sum()


    def calculate_clip_score(self, caption_list, image_unprocessed):
        image_unprocessed = 0.5 * (image_unprocessed + 1.)
        image_unprocessed.clamp_(0., 1.)
        img_resize = Resize((224))(image_unprocessed)
        return self.get_clip_score(img_resize,caption_list)