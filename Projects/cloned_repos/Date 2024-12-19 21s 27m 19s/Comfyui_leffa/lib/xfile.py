import folder_paths
import os
import base64
import numpy as np
from PIL import Image,ImageOps, ImageFilter

import io
import glob

import shutil

comfy_path = os.path.dirname(folder_paths.__file__)
custom_nodes_path = os.path.join(comfy_path, "custom_nodes")

# D:\comfyui\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_CXH_ALY
# current_folder = os.path.dirname(os.path.abspath(__file__))

# 节点路径
def node_path(node_name):
    return os.path.join(custom_nodes_path,node_name)

# 创建文件夹
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径

# 获取所有图片文件路径
def get_all_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def list_files_names(input_dir, ext):  
    # 确保目录存在，如果不存在则创建  
    if not os.path.exists(input_dir):  
        os.makedirs(input_dir)  
      
    # 使用glob查找指定扩展名的文件  
    file_paths = glob.glob(os.path.join(input_dir, '*' + ext))  
      
    # 初始化一个空列表来存储文件名（不带扩展名）  
    file_names = []  
      
    # 遍历文件路径，提取文件名并去除扩展名  
    for file_path in file_paths:  
        # os.path.splitext() 返回文件名和扩展名，我们只需要文件名部分  
        filename, _ = os.path.splitext(os.path.basename(file_path))  
        file_names.append(filename)  
      
    # 返回文件名列表  
    return file_names 

def remove_empty_lines(text):
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(non_empty_lines)

def clear_folder(folder_path):
    """
    清空指定文件夹中的所有内容。
    
    :param folder_path: 要清空的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在。")
        return
    
    # 遍历文件夹中的所有文件和子文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # 如果是文件或符号链接，则删除
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # 如果是文件夹，则删除整个文件夹及其内容
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'删除 {file_path} 时出错: {e}')
    
    print(f"文件夹 {folder_path} 已清空。")