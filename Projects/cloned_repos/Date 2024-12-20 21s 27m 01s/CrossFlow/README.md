# CrossFlow
This is a PyTorch-based reimplementation of CrossFlow, as proposed in 

>  Flowing from Words to Pixels: A Framework for Cross-Modality Evolution
>
>  [Qihao Liu](https://qihao067.github.io/) | [Xi Yin](https://xiyinmsu.github.io/) | [Alan Yuille](https://cogsci.jhu.edu/directory/alan-yuille/) | [Andrew Brown](https://www.robots.ox.ac.uk/~abrown/) | [Mannat Singh](https://ai.meta.com/people/1287460658859448/mannat-singh/)
>
>  [[project page](https://cross-flow.github.io/)] | [[paper](https://arxiv.org/pdf/2412.15213)] | [[arxiv](https://arxiv.org/abs/2412.15213)]

![teaser](https://github.com/qihao067/CrossFlow/blob/main/imgs/teaser.jpg)

This repository provides a **PyTorch-based reimplementation** of CrossFlow for the text-to-image generation task, with the following differences compared to the original paper:

- **Model Architecture**: The original paper utilizes DiMR as the model architecture. In contrast, this codebase supports training and inference with both [DiT](https://github.com/facebookresearch/DiT) (ICCV 2023, a widely adopted architecture) and [DiMR](https://github.com/qihao067/DiMR) (NeurIPS 2024, a state-of-the-art architecture).
- **Dataset**: The original model was trained on proprietary 350M dataset. In this implementation, the models are trained on open-source datasets, including [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) and [JourneyDB](https://journeydb.github.io/).
- **LLMs**: The original model only supports CLIP as the language model, whereas this implementation includes models with CLIP and T5-XXL.

______

## TODO

- [x] Release inference code and 512px CLIP DiMR-based model.
- [ ] Release training code and a detailed training tutorial (ETA: Dec 20).
- [ ] Release inference code for linear interpolation and arithmetic.
- [ ] Release all pretrained checkpoints, including:   (ETA: Dec 23)
  - 256px CLIP DiMR-based model, 
  - 256px T5XXL DiMR-based model, 
  - 512px T5XXL DiMR-based model, 
  - 512px T5XXL DiT-based model
- [ ] Provide a demo via Hugging Face Space and Colab.

______

## Requirements

The code has been tested with PyTorch 2.1.2 and Cuda 12.1.

An example of installation commands is provided as follows:

```
git clone git@github.com:qihao067/CrossFlow.git
cd CrossFlow

pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip3 install -U --pre triton
pip3 install -r requirements.txt
```

------

## Pretrained Models

| Architecture | Resolutions | LM     | Training Epochs    | #Params. (FM) | Download                                                     |
| :----------- | ----------- | ------ | ------------------ | ------------- | ------------------------------------------------------------ |
| DiMR         | 256x256     | CLIP   |                    | 0.95B         | (TODO)                                                       |
| DiMR         | 256x256     | T5-XXL |                    | 0.95B         | (TODO)                                                       |
| DiMR         | 512x512     | CLIP   | 1 (LIAON)+10 (JDB) | 0.98B         | [[link](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_512px_clip_dimr.pth)] |
| DiMR         | 512x512     | T5-XXL |                    |               | (TODO)                                                       |
| DiT          | 512x512     | T5-XXL |                    |               | (TODO)                                                       |

------

## Sampling

You can sample from the pre-trained CrossFLow model with the [`demo_t2i.py`](https://github.com/qihao067/CrossFlow/blob/main/demo_t2i.py). Before running the script, download the appropriate checkpoint and configure hyperparameters such as the classifier-free guidance scale, random seed, and mini-batch size in the corresponding configuration files.

To accelerate the sampling process, the script supports multi-GPU sampling. For example, to sample from the 512px CLIP DiMR-based CrossFlow model with `N` GPUs, you can use the following command. It generates `N x mini-batch size` images each time:

```
accelerate launch --multi_gpu --num_processes N --mixed_precision bf16 demo_t2i.py \
          --config=configs/t2i_512px_clip.py \
          --nnet_path=path/to/t2i_512px_clip_dimr.pth \
          --img_save_path=temp_saved_images \
          --prompt='A dog flying in the sky' \
```



------

## Terms of use

The project is created for research purposes.

______

## 
