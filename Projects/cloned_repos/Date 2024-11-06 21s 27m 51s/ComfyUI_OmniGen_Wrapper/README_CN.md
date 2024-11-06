# ComfyUI_OmniGen_Wrapper

![image](image/omnigen_wrapper_example.jpg)    
本节点是 [OmniGen](https://github.com/VectorSpaceLab/OmniGen) 项目的非官方封装，在ComfyUI中运行。    
量化代码参考自 [Manni1000/OmniGen](https://github.com/Manni1000/OmniGen)。

### 安装插件
在ComfyUI/custom_nodes文件夹打开终端窗口，输入以下命令：
```
git clone https://github.com/chflame163/ComfyUI_OmniGen_Wrapper.git
```

### 安装依赖
请在ComfyUI的Python 环境里运行以下命令：
```
python -s -m pip install -r ComfyUI/custom_nodes/ComfyUI_OmniGen_Wrapper/requirements.txt
```

### 下载模型
首次运行插件时将自动下载模型。也可以手动下载，在下面两个下载途径选择其一:    

从Huggingface下载:
* 从 [Shitao/OmniGen-v1](https://huggingface.co/Shitao/OmniGen-v1/tree/main) 下载全部模型文件，并复制到```ComfyUI/models/OmniGen/Shitao/OmniGen-v1```文件夹；    
* 从 [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae/tree/main) 下载 diffusion_pytorch_model.safetensors 和 config.json 两个文件，并复制到```ComfyUI/models/OmniGen/Shitao/OmniGen-v1/vae```文件夹。    

或者从百度网盘下载全部模型文件并复制到```ComfyUI/models/OmniGen/Shitao/OmniGen-v1```文件夹:
* [百度网盘](https://pan.baidu.com/s/1uivyo_voaZ668nT3aMLw8Q?pwd=ma06)


### 使用节点
启动ComfyUI，点击右键菜单-```Add Node``` - ```😺dzNodes``` - ```OmniGen Wrapper```，找到节点。    
![image](image/add_node.jpg)   
或者在节点搜索栏中输入 OmniGen Wrapper 找到节点。    
![image](image/search_node.jpg)

### 节点参数说明
![image](image/omnigen_wrapper_node.jpg)

* image_1: 可选输入图片1。如果输入，须在prompt中描述此图，用```{image_1}```指代。
* image_2: 可选输入图片2。如果输入，须在prompt中描述此图，用```{image_2}```指代。
* image_3: 可选输入图片3。如果输入，须在prompt中描述此图，用```{image_3}```指代。
* dtype: 模型精度，default为模型默认精度, 可选int8。默认精度大约占用12GB显存，int8大约占用7GB显存。
* prompt: 生成图片的提示词。如果有图片输入，请用```{image_1}```、```{image_2}```、```{image_3}```指代。
* width: 生成图片的宽度，必须为16的倍数。
* height: 生成图片的高度，必须为16的倍数。
* guidance_scale: 引导比例。较高的值会使模型的生成结果更倾向于条件，但可能损失图像的多样性和自由度。
* image_guidance_scale: 图片引导比例。
* steps: 图片生成推理步数。
* separate_cfg_infer: 在不同引导下分别对图像进行推理；这可以在生成大尺寸图像时节省内存，但会使推理速度变慢。
* use_kv_cache: 使用kv缓存以加快推理速度。
* seed: 随机种子。推理使用不同的种子产生不同结果。
* control_after_generate: 每次运行时种子值变化选项。
* cache_model: 设置为True时缓存模型，下次运行无需再次加载模型。

### Star 记录 

[![Star History Chart](https://api.star-history.com/svg?repos=chflame163/ComfyUI_OmniGen_Wrapper&type=Date)](https://star-history.com/#chflame163/ComfyUI_OmniGen_Wrapper&Date)

###  声明
本节点遵照MIT开源协议，有部分功能代码和模型来自其他开源项目，感谢原作者。如果作为商业用途，请查阅原项目授权协议使用。
