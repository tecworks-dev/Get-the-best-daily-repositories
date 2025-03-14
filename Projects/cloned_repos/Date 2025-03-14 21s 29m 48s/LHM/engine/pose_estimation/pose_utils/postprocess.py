import time
import torch
import torch.nn.functional as F
import numpy as np

def get_gaussian_kernel_1d(kernel_size, sigma, device):
    x = torch.arange(kernel_size).float() - (kernel_size // 2)
    g = torch.exp(-((x ** 2) / (2 * sigma ** 2)))
    g /= g.sum()

    kernel_weight = g.view(1, 1, -1).to(device)


    return kernel_weight

def gaussian_filter_1d(data, kernel_size=3, sigma=1.0, weight=None):
    kernel_weight = get_gaussian_kernel_1d(kernel_size, sigma, data.device) if weight is None else weight
    data = F.pad(data, (kernel_size // 2, kernel_size // 2), mode='replicate')
    return F.conv1d(data, kernel_weight)


def exponential_smoothing(x, d_x, alpha=0.5):
    return d_x + alpha * (x - d_x)


class OneEuroFilter:
    # param setting:
    #   realtime v2m: min_cutoff=1.0, beta=1.5
    #   motionshop 2d keypoint: min_cutoff=1.7, beta=0.3
    def __init__(self, min_cutoff=1.0, beta=0.0, sampling_rate=30, d_cutoff=1.0,  device='cuda'):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.sampling_rate = sampling_rate
        self.x_prev = None
        self.dx_prev = None
        self.d_cutoff = d_cutoff
        self.pi = torch.tensor(torch.pi, device=device)

    def smoothing_factor(self, cutoff):
        
        r = 2 * self.pi * cutoff / self.sampling_rate
        return r/ (1 + r)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = torch.zeros_like(x)
            return x

        
        a_d = self.smoothing_factor(self.d_cutoff)
        # 计算当前的速度
        dx = (x - self.x_prev) * self.sampling_rate

        dx_hat = exponential_smoothing(dx, self.dx_prev, a_d)

        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)
        a = self.smoothing_factor(cutoff)

        x_hat = exponential_smoothing(x, self.x_prev, a)

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


class Filter():
    filter_factory = {
        'gaussian': get_gaussian_kernel_1d,
    }

    def __init__(self, target_data, filter_type, filter_args):
        self.target_data = target_data
        self.filter = self.filter_factory[filter_type]
        self.filter_args = filter_args

    def process(self, network_outputs):
        filter_data = []
        for human in network_outputs:
            filter_data.append(human[self.target_data])
        filter_data = torch.stack(filter_data, dim=0)
        
        filter_data = self.filter(filter_data, **self.filter_args)
        
        for i, human in enumerate(network_outputs):
            human[self.target_data] = filter_data[i]


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    from rot6d import rotation_6d_to_axis_angle, axis_angle_to_rotation_6d

    from humans import get_smplx_joint_names
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    data_types = ['rotvec']#, 'j3d']
    observe_keypoints = ['pelvis', 'head', 'left_wrist', 'left_knee']
    joint_names = get_smplx_joint_names()

    
    data = np.load(f'{args.data_path}/shape_{args.name}.npy')
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    for i in range(2):
        for j in range(2):
            x = data[:, i*4 + j*2]
            print(x.shape)
            axs[i, j].plot(x)

            axs[i, j].set_title(f'{4 * i + 2 * j}')
    axs[i, j].plot(np.load(f'{args.data_path}/dist_{args.name}.npy'))
    plt.tight_layout()
    plt.savefig(f'{args.save_path}/shape_{args.name}.jpg')
    # for data_type in data_types:
    #     data = np.load(f'{args.data_path}/{data_type}_{args.name}.npy')
    #     fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    #     for i in range(2):
    #         for j in range(2):
    #             # todo: something wrong here
    #             filter = OneEuroFilter(min_cutoff=1, beta=0.01, sampling_rate=30, device='cuda:0')
    #             x = data[:, joint_names.index(observe_keypoints[i*2+j])] #(F, 3)
    #             print(x.shape)

    #             x = axis_angle_to_rotation_6d(torch.tensor(x, device='cuda:0'))
                
    #             x_filtered = x.clone()
    #             start = time.time()
    #             for k in range(x.shape[0]):
    #                 x_filtered[k] = filter.filter(x[k])

    #             print(x_filtered.shape[0]/(time.time()-start))
    #             # x_filtered = x.clone()
    #             # a = 0.5
    #             # for k in range(1, x.shape[0]):
    #             #     x_filtered[k] = (1 - a) * x_filtered[k-1] + a * x[k]
    #             #theta = np.linalg.norm(x, axis=-1)
    #             #x = x / theta[..., None]
                
                
    #             # f, n = x.shape
    #             # x_filtered = gaussian_filter_1d(x.permute(1, 0).view(n, 1, -1), 11, 11)
    #             # x_filtered = x_filtered.view(n, -1).permute(1, 0)
                
    #             x = rotation_6d_to_axis_angle(x).cpu().numpy()
    #             x_filtered  = rotation_6d_to_axis_angle(x_filtered).cpu().numpy()
    #             axs[i, j].plot(x[..., 0])
    #             axs[i, j].plot(x[..., 1])
    #             axs[i, j].plot(x[..., 2])

    #             axs[i, j].plot(x_filtered[..., 0])
    #             axs[i, j].plot(x_filtered[..., 1])
    #             axs[i, j].plot(x_filtered[..., 2])
    #             #axs[i, j].plot(theta)

    #             axs[i, j].set_title(f'{observe_keypoints[i*2 + j]}')
    #     plt.tight_layout()
    #     plt.savefig(f'{args.save_path}/{data_type}_{args.name}.jpg')
    