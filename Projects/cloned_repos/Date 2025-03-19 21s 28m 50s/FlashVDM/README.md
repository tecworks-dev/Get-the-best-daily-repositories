<div align="center">
    <img src="https://github.com/user-attachments/assets/96a9c182-6ac9-4744-89d2-9f95aa1e7b67"  height=120>
</div>
</br>

⚡️⚡️ Try it Now with **[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)** for super fast high-quality shape generation within 1 second on 4090.




https://github.com/user-attachments/assets/a2cbc5b8-be22-49d7-b1c3-7aa2b20ba460





<div align="center">
  <a href=https://huggingface.co/spaces/tencent/Hunyuan3D-2mini-Turbo  target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/tencent/Hunyuan3D-2mini target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href="assets/report.pdf" target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/txhunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
</div>


## What is FlashVDM?
FlashVDM is a general framework for accelerating shape generation Vectset Diffusion Model (VDM), such as [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2), [Michelangelo](https://github.com/NeuralCarver/Michelangelo), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D), [CLAY](https://github.com/CLAY-3D/OpenCLAY), [TripoSG](https://arxiv.org/abs/2502.06608), [Dora](https://github.com/Seed3D/Dora) and etc.

It features two techniques for both VAE and DiT acceleration: 

1. ***Lightning Vectset Decoder*** that drastically lowers decoding FLOPs without any loss in decoding quality, achieving over **45x speedup**.
2. ***Progressive Flow Distillation*** that enables flexible diffusion sampling with as few as **5 inference steps** and comparable quality.

<img src="https://github.com/user-attachments/assets/bcc1f43e-4cfa-47f3-9a45-421f75cf5138"  height=250>

## How to Use?

Visit **[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)** to access the integration of FlashVDM with Hunyuan3D-2.

```diff
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
-   subfolder='hunyuan3d-dit-v2-0',
+   subfolder='hunyuan3d-dit-v2-0-turbo',
    use_safetensors=True,
)
+pipeline.enable_flashvdm()

pipeline(
    image=image,
-   num_inference_steps=50,
+   num_inference_steps=5,
)[0]
```

## Supported Models

Hunyuan3D-2 series

| Model                    | Description                 | Date       | Size | Huggingface                                                                               |
| ------------------------ | --------------------------- | ---------- | ---- | ----------------------------------------------------------------------------------------- |
| Hunyuan3D-DiT-v2-0-Fast  | Guidance Distillation Model | 2025-02-03 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-fast)  |
| Hunyuan3D-DiT-v2-0-Turbo | Step Distillation Model     | 2025-03-15 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-turbo) |


Hunyuan3D-2mini series

| Model                       | Description                 | Date       | Size | Huggingface                                                                                      |
| --------------------------- | --------------------------- | ---------- | ---- | ------------------------------------------------------------------------------------------------ |
| Hunyuan3D-DiT-v2-mini-Fast  | Guidance Distillation Model | 2025-02-03 | 0.6B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini-fast)  |
| Hunyuan3D-DiT-v2-mini-Turbo | Step Distillation Model     | 2025-03-15 | 0.6B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini-turbo) |


Hunyuan3D-2mv series

| Model                     | Description                 | Date       | Size | Huggingface                                                                                  |
| ------------------------- | --------------------------- | ---------- | ---- | -------------------------------------------------------------------------------------------- |
| Hunyuan3D-DiT-v2-mv-Fast  | Guidance Distillation Model | 2025-03-19 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv-fast)  |
| Hunyuan3D-DiT-v2-mv-Turbo | Step Distillation Model     | 2025-03-19 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv-turbo) |


## Citation

If you found this repository helpful, please cite our report:

```bibtex
@misc{lai2025flashvdm,
    title={Unleashing Vectset Diffusion Model for Fast Shape Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
}
```
