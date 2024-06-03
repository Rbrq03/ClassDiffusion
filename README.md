# ClassDiffusion

Official imple. of ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance

---

**[ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance](https://arxiv.org/pdf/2405.17532)**

[Jiannan Huang](https://rbrq03.github.io/), [Jun Hao Liew](https://scholar.google.com.sg/citations?user=8gm-CYYAAAAJ), [Hanshu Yan](https://hanshuyan.github.io), [Yuyang Yin](https://yuyangyin.github.io), [Yao Zhao](http://mepro.bjtu.edu.cn/zhaoyao/index.htm), [Yunchao Wei](https://weiyc.github.io/index.html)

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://classdiffusion.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.17532-b31b1b.svg)](https://arxiv.org/pdf/2405.17532)

<p align="center">
<img src="assert/img/story.jpg" width="1080px"/>  
<br>
<em>Our method can generate more aligned personalized images with explicit class guidance</em>
</p>

## News

- [3 Jun. 2024] Code Released!
- [29 May. 2024] Paper Released!

## Code Usage

**Set Up**

```
git clone https://github.com/Rbrq03/ClassDiffusion.git
cd ClassDiffusion
pip install -r requirements.txt
```

_Warning:Currently, ClassDiffusion don't support peft, please ensure peft is uninstalled in your environment, or check [PR](https://github.com/huggingface/diffusers/pull/7272). We will move forward with this PR merge soon._

### Training

**Single Concept**

```
bash scripts/train_single.sh
```

**Multiple Concepts**

```
bash scripts/train_multi.sh
```

### Inference

**single concept**

```
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
).to("cuda")
pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
pipeline.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")

image = pipeline(
    "<new1> dog swimming in the pool",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("dog.png")
```

**Multiple Concepts**

```
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
pipeline.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")
pipeline.load_textual_inversion("path-to-save-model", weight_name="<new2>.bin")

image = pipeline(
    "a <new1> teddy bear sitting in front of a <new2> barn",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("multi-subject.png")
```

<!-- **BLIP2-T** -->

## Results

**Single Concept Results**

<p align="center">
<img src="assert/img/singleconcept.png" width="1080px"/>  
<br>
</p>

**Multiple Concepts Results**

<p align="center">
<img src="assert/img/multiple.png" width="1080px"/>  
<br>
<em></em>
</p>

## TODO

- [x] Training Code for ClassDiffusion
- [x] Inference Code for ClassDiffusion
- [ ] Pipeline for BLIP2-T Score
- [ ] Inference Code for Video Generation with ClassDiffusion

## Citation

If you make use of our work, please cite our paper.

```bibtex
@article{huang2024classdiffusion,
  title={ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance},
  author={Huang, Jiannan and Liew, Jun Hao and Yan, Hanshu and Yin, Yuyang and Zhao, Yao and Wei, Yunchao},
  journal={arXiv preprint arXiv:2405.17532},
  year={2024}
}
```

## Acknowledgement

We thanks to the following repo for their excellent and well-documented code based:

- [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
- [https://github.com/adobe-research/custom-diffusion](https://github.com/adobe-research/custom-diffusion)
- [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- [https://github.com/google/dreambooth](https://github.com/google/dreambooth)
