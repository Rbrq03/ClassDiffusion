# import torch

import types
import torch
from utils.load_attn_weight import load_custom_attn_param

from diffusers import (
    AnimateDiffPipeline,
    DDIMScheduler,
    MotionAdapter,
    DiffusionPipeline,
)
from diffusers.utils import export_to_gif

model_id = "runwayml/stable-diffusion-v1-5"
ckpt_path = "your_ckpt_path"  # TODO: modify here

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

pipeline.unet.load_custom_attn_param = types.MethodType(
    load_custom_attn_param, pipeline.unet
)
pipeline.unet.load_custom_attn_param(
    ckpt_path,
    weight_name="pytorch_custom_diffusion_weights.bin",
)
pipeline.load_textual_inversion(ckpt_path, weight_name="<new1>.bin")

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16
)
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    unet=pipeline.unet,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer,
    torch_dtype=torch.float16,
)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=("<new1> dog running on the street"),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
)
frames = output.frames[0]
export_to_gif(frames, "test.gif", fps=4)
