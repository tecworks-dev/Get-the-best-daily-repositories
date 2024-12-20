import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import yaml
import numpy as np
from PIL import Image
import rembg
import importlib
import torch
import tempfile
import json
#import spaces
from core.models import DiT_models
from core.diffusion import create_diffusion
from core.utils.dinov2 import Dinov2Model
from core.utils.math_utils import unnormalize_params

from huggingface_hub import hf_hub_download

# Setup PyTorch:
device = torch.device('cuda')

# Define the cache directory for model files
#model_cache_dir = './ckpts/'
#os.makedirs(model_cache_dir, exist_ok=True)

# load generators & models
generators_choices = ["chair", "table", "vase", "basket", "flower", "dandelion"]
factory_names = ["ChairFactory", "TableDiningFactory", "VaseFactory", "BasketBaseFactory", "FlowerFactory", "DandelionFactory"]
generator_path = "./core/assets/"
generators, configs, models = [], [], []
for category, factory in zip(generators_choices, factory_names):
    # load generator
    module = importlib.import_module(f"core.assets.{category}")
    gen = getattr(module, factory)
    generator = gen(0)
    generators.append(generator)
    # load configs
    config_path = f"./configs/demo/{category}_demo.yaml"
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    configs.append(cfg)
    # load models
    latent_size = cfg["num_params"]
    model = DiT_models[cfg["model"]](input_size=latent_size).to(device)
    # load a custom DiT checkpoint from train.py:
    # download the checkpoint if not found:
    if not os.path.exists(cfg["ckpt_path"]):
        model_dir, model_name = os.path.dirname(cfg["ckpt_path"]), os.path.basename(cfg["ckpt_path"])
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = hf_hub_download(repo_id="TencentARC/DI-PCG", 
                            local_dir=model_dir, filename=model_name)
        print("Downloading checkpoint {} from Hugging Face Hub...".format(model_name))
    print("Loading model from {}".format(cfg["ckpt_path"]))
    
    state_dict = torch.load(cfg["ckpt_path"], map_location=lambda storage, loc: storage)
    if "ema" in state_dict:  # supports checkpoints from train.py
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
    
diffusion = create_diffusion(str(cfg["num_sampling_steps"]))
# feature model
feature_model = Dinov2Model()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background):
    # resize
    if input_image.size[0] != 256 or input_image.size[1] != 256:
        input_image = input_image.resize((256, 256))
    # remove background
    if do_remove_background:
        processed_image = rembg.remove(np.array(input_image))
    # white background
    else:
        processed_image = input_image
    return processed_image

#@spaces.GPU
def sample(image, seed, category):
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # generator & model
    idx = generators_choices.index(category)
    generator, cfg, model = generators[idx], configs[idx], models[idx]
    
    # encode condition image feature
    # convert RGBA images to RGB, white background
    input_image_np = np.array(image)
    mask = input_image_np[:, :, -1:] > 0
    input_image_np = input_image_np[:, :, :3] * mask + 255 * (1 - mask)
    image = input_image_np.astype(np.uint8)
    
    img_feat = feature_model.encode_batch_imgs([np.array(image)], global_feat=False)

    # Create sampling noise:
    latent_size = int(cfg['num_params'])
    z = torch.randn(1, 1, latent_size, device=device)
    y = img_feat

    # No classifier-free guidance:
    model_kwargs = dict(y=y)

    # Sample target params:
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples = samples[0].squeeze(0).cpu().numpy()

    # unnormalize params
    params_dict = generator.params_dict
    params_original = unnormalize_params(samples, params_dict)
    
    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".glb", delete=False).name
    params_fpath = tempfile.NamedTemporaryFile(suffix=f".npy", delete=False).name
    np.save(params_fpath, params_original)
    print(mesh_fpath)
    print(params_fpath)
    # generate 3D using sampled params - TODO: this is a hacky way to go through PCG pipeline, avoiding conflict with gradio
    command = f"python ./scripts/generate.py --config ./configs/demo/{category}_demo.yaml --output_path {mesh_fpath} --seed {seed} --params_path {params_fpath}"
    os.system(command)
    
    return mesh_fpath


import gradio as gr

_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/DI-PCG' target='_blank'><b>DI-PCG: Diffusion-based Efficient Inverse Procedural Content Generation for High-quality 3D Asset Creation</b></a></h2>

**DI-PCG** is a diffusion model which directly generates a procedural generator's parameters from a single image, resulting in high-quality 3D meshes.

Code: <a href='https://github.com/TencentARC/DI-PCG' target='_blank'>GitHub</a>. Techenical report: <a href='' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- DI-PCG trains a diffusion model for each procedural generator. Current supported generators are: Chair, Table, Vase, Basket, Flower, Dandelion from <a href="https://github.com/princeton-vl/infinigen">Infinigen</a>.
- The diversity of the generated meshes are strictly bounded by the procedural generators. For out-of-domain shapes, DI-PCG may only provide closest approximations.
'''

_CITE_ = r"""
If DI-PCG is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/DI-PCG' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/DI-PCG?style=social)](https://github.com/TencentARC/DI-PCG)
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex

```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file]() for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b></b>.
"""
def update_examples(category):
    samples = [[os.path.join(f"examples/{category}", img_name)]
        for img_name in sorted(os.listdir(f"examples/{category}"))]
    print(samples)
    return gr.Dataset(samples=samples)

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            # select the generator category
            with gr.Row():
                with gr.Group():
                    generator_category = gr.Radio(
                        choices=[
                            "chair",
                            "table",
                            "vase",
                            "basket",
                            "flower",
                            "dandelion",
                        ],
                        value="chair",
                        label="category",
                    )
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGB",
                    sources='upload',
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=False
                    )
                    sample_seed = gr.Number(value=0, label="Seed Value", precision=0)

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                examples = gr.Examples(
                    [os.path.join(f"examples/chair", img_name) for img_name in sorted(os.listdir(f"examples/chair"))],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=5
                )
                generator_category.change(update_examples, generator_category, outputs=examples.dataset)

        with gr.Column():
            with gr.Row():
                with gr.Tab("Geometry"):
                    output_model_obj = gr.Model3D(
                        label="Output Model",
                        #width=768,
                        display_mode="wireframe",
                        interactive=False
                    )
                #with gr.Tab("Textured"):
                #    output_model_obj = gr.Model3D(
                #        label="Output Model (STL Format)",
                #        #width=768,
                #        interactive=False,
                #    )
                #    gr.Markdown("Note: Texture and Material are randomly assigned by the procedural generator.")


    gr.Markdown(_CITE_)
    mv_images = gr.State()

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=sample,
        inputs=[processed_image, sample_seed, generator_category],
        outputs=[output_model_obj],
    )

demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=43839)