 # 计算最接近的2的次幂
def nearest_power_of_two(x):
    return 2 ** (x-1).bit_length()

def preprocess_image(img, target_width=136):
    """
    预处理图像，确保宽度为2的倍数

    Args:
        image_path: 图像路径
        target_width: 目标宽度

    Returns:
        预处理后的图像张量
    """
    width, height = img.size

    # 如果宽度不是2的倍数，则填充
    if width % 2 != 0:
        padding = target_width - width
        img = img.pad((0, padding, 0, 0), padding_mode='constant', value=0)

    # 转换为PyTorch张量
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()

    # 归一化等其他预处理操作

    return img_tensor