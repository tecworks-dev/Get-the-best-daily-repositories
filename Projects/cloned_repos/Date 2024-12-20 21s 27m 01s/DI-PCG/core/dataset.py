import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
from core.utils.io import read_list_from_txt
from core.utils.math_utils import normalize_params

class ImageParamsDataset(Dataset):
    def __init__(self, data_root, list_file, params_dict_file):
        self.data_root = data_root
        self.data_lists = read_list_from_txt(os.path.join(data_root, list_file))
        self.params_dict = json.load(open(os.path.join(data_root, params_dict_file), 'r'))

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, idx):
        name = self.data_lists[idx]
        id = name.split("/")[0]
        params = json.load(open(os.path.join(self.data_root, id, "params.txt"), 'r'))
        # normalize the params to [-1, 1] range for training diffusion
        normalized_params = normalize_params(params, self.params_dict)
        normalized_params_values = np.array(list(normalized_params.values()))
        img = cv2.cvtColor(cv2.imread(os.path.join(self.data_root, name)), cv2.COLOR_BGR2RGB)
        
        img_feat_name = os.path.join(self.data_root, name.replace(".png", "_dino_token.npy"))
        if not os.path.exists(img_feat_name):
            img_feat_file = np.load(os.path.join(self.data_root, name.replace(".png", "_dino_token.npz")))
            img_feat = img_feat_file['arr_0']
            img_feat_file.close()
        else:
            img_feat = np.load(img_feat_name)
        img_feat_t = torch.from_numpy(img_feat).float()
        return torch.from_numpy(normalized_params_values).float(), img_feat_t, img

        