# ComfyUI wrapper nodes for [Mochi](https://github.com/genmoai/models) video generator


# WORK IN PROGRESS

https://github.com/user-attachments/assets/a714b70f-dcdb-4f91-8a3d-8da679a28d6e


Can use flash_attn, pytorch attention (sdpa) or [sage attention](https://github.com/thu-ml/SageAttention), sage being fastest.

Depending on frame count can fit under 20GB, VAE decoding is heavy and there is experimental tiled decoder (taken from CogVideoX -diffusers code) which allows higher frame counts, so far highest I've done is 97 with the default tile size 2x2 grid.

Models:

https://huggingface.co/Kijai/Mochi_preview_comfy/tree/main

model to: `ComfyUI/models/diffusion_models/mochi`

vae to: `ComfyUI/models/vae/mochi`

There is autodownload node (also will be normal loader node)
