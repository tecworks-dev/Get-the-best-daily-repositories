import json
import os
import tempfile
import time

import click
import numpy as np
#import ray
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from mochi_preview.t2v_synth_mochi import T2VSynthMochiModel

model = None
model_path = "weights"
def noexcept(f):
    try:
        return f()
    except:
        pass
# class MochiWrapper:
#     def __init__(self, *, num_workers, **actor_kwargs):
#         super().__init__()
#         RemoteClass = ray.remote(T2VSynthMochiModel)
#         self.workers = [
#             RemoteClass.options(num_gpus=1).remote(
#                 device_id=0, world_size=num_workers, local_rank=i, **actor_kwargs
#             )
#             for i in range(num_workers)
#         ]
#         # Ensure the __init__ method has finished on all workers
#         for worker in self.workers:
#             ray.get(worker.__ray_ready__.remote())
#         self.is_loaded = True

#     def __call__(self, args):
#         work_refs = [
#             worker.run.remote(args, i == 0) for i, worker in enumerate(self.workers)
#         ]

#         try:
#             for result in work_refs[0]:
#                 yield ray.get(result)

#             # Handle the (very unlikely) edge-case where a worker that's not the 1st one
#             # fails (don't want an uncaught error)
#             for result in work_refs[1:]:
#                 ray.get(result)
#         except Exception as e:
#             # Get exception from other workers
#             for ref in work_refs[1:]:
#                 noexcept(lambda: ray.get(ref))
#             raise e
        
def set_model_path(path):
    global model_path
    model_path = path


def load_model():
    global model, model_path
    if model is None:
        #ray.init()
        MOCHI_DIR = model_path
        VAE_CHECKPOINT_PATH = f"{MOCHI_DIR}/mochi_preview_vae_bf16.safetensors"
        MODEL_CONFIG_PATH = f"{MOCHI_DIR}/dit-config.yaml"
        MODEL_CHECKPOINT_PATH = f"{MOCHI_DIR}/mochi_preview_dit_fp8_e4m3fn.safetensors"

        model = T2VSynthMochiModel(
            device_id=0,
            world_size=1,
            local_rank=0,
            vae_stats_path=f"{MOCHI_DIR}/vae_stats.json",
            vae_checkpoint_path=VAE_CHECKPOINT_PATH,
            dit_config_path=MODEL_CONFIG_PATH,
            dit_checkpoint_path=MODEL_CHECKPOINT_PATH,
        )

def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
    const = quadratic_coef * (linear_steps ** 2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule

def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
):
    load_model()

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "mochi_args": {
            "sigma_schedule": sigma_schedule,
            "cfg_schedule": cfg_schedule,
            "num_inference_steps": num_inference_steps,
            "batch_cfg": False,
        },
        "prompt": [prompt],
        "negative_prompt": [negative_prompt],
        "seed": seed,
    }

    final_frames = None
    for cur_progress, frames, finished in tqdm(model.run(args, stream_results=True), total=num_inference_steps + 1):
        final_frames = frames

    assert isinstance(final_frames, np.ndarray)
    assert final_frames.dtype == np.float32

    final_frames = rearrange(final_frames, "t b h w c -> b t h w c")
    final_frames = final_frames[0]

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"output_{int(time.time())}.mp4")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(final_frames):
            frame = (frame * 255).astype(np.uint8)
            frame_img = Image.fromarray(frame)
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)
            frame_paths.append(frame_path)

        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        ffmpeg_cmd = f"ffmpeg -y -r 30 -i {frame_pattern} -vcodec libx264 -pix_fmt yuv420p {output_path}"
        os.system(ffmpeg_cmd)

        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(args, f, indent=4)

    return output_path


@click.command()
@click.option("--prompt", default="""
              a high-motion drone POV flying at high speed through a vast desert environment, with dynamic camera movements capturing sweeping sand dunes, 
              rocky terrain, and the occasional dry brush. The camera smoothly glides over the rugged landscape, weaving between towering rock formations and 
              diving low across the sand. As the drone zooms forward, the motion gradually slows down, shifting into a close-up, hyper-detailed shot of a spider 
              resting on a sunlit rock. The scene emphasizes cinematic motion, natural lighting, and intricate texture details on both the rock and the spider’s body, 
              with a shallow depth of field to focus on the fine details of the spider’s legs and the rough surface beneath it. The atmosphere should feel immersive and alive, 
              with the wind subtly blowing sand grains across the frame."""
              , required=False, help="Prompt for video generation.")
@click.option(
    "--negative_prompt", default="", help="Negative prompt for video generation."
)
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=163, type=int, help="Number of frames.")
@click.option("--seed", default=12345, type=int, help="Random seed.")
@click.option("--cfg_scale", default=4.5, type=float, help="CFG Scale.")
@click.option(
    "--num_steps", default=64, type=int, help="Number of inference steps."
)
@click.option("--model_dir", required=True, help="Path to the model directory.")
def generate_cli(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_steps,
    model_dir,
):
    set_model_path(model_dir)
    output = generate_video(
        prompt,
        negative_prompt,
        width,
        height,
        num_frames,
        seed,
        cfg_scale,
        num_steps,
    )
    click.echo(f"Video generated at: {output}")

if __name__ == "__main__":
    generate_cli()
