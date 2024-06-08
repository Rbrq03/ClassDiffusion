import torch
import warnings
import safetensors
from typing import Callable, Dict, List, Optional, Union

from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


def load_custom_attn_param(
    self,
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    **kwargs
):
    r"""
    Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
    defined in
    [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
    and be a `torch.nn.Module` class.

    Parameters:
        pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
            Can be either:

                - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                    the Hub.
                - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                    with [`ModelMixin.save_pretrained`].
                - A [torch state
                    dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
            is not used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
            incompletely downloaded files are deleted.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only load local model weights and configuration files or not. If set to `True`, the model
            won't be downloaded from the Hub.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
            `diffusers-cli login` (stored in `~/.huggingface`) is used.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
            Speed up model loading only loading the pretrained weights and not initializing the weights. This also
            tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
            Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
            argument to `True` will raise an error.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
            allowed by Git.
        subfolder (`str`, *optional*, defaults to `""`):
            The subfolder location of a model file within a larger model repository on the Hub or locally.
        mirror (`str`, *optional*):
            Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
            guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
            information.

    Example:

    ```py
    from diffusers import AutoPipelineForText2Image
    import torch

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    ).to("cuda")
    pipeline.unet.load_attn_procs(
        "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
    )
    ```
    """
    from diffusers.models.attention_processor import CustomDiffusionAttnProcessor
    from diffusers.models.lora import (
        LoRACompatibleConv,
        LoRACompatibleLinear,
        LoRAConv2dLayer,
        LoRALinearLayer,
    )

    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
    # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    network_alphas = kwargs.pop("network_alphas", None)

    _pipeline = kwargs.pop("_pipeline", None)
    use_custom_diffusion = kwargs.pop("custom_diffusion", None)

    is_network_alphas_none = network_alphas is None

    allow_pickle = False

    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        # Let's first try to load .safetensors weights
        if (use_safetensors and weight_name is None) or (
            weight_name is not None and weight_name.endswith(".safetensors")
        ):
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = safetensors.torch.load_file(model_file, device="cpu")
            except IOError as e:
                if not allow_pickle:
                    raise e
                # try loading non-safetensors weights
                pass
        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")
    else:
        state_dict = pretrained_model_name_or_path_or_dict

    # is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())
    is_custom_diffusion = (
        any("custom_diffusion" in k for k in state_dict.keys()) or use_custom_diffusion
    )

    assert is_custom_diffusion == True

    unet_state_dict = self.state_dict()

    for name, weight in state_dict.items():
        if name.endswith("weight"):
            name = name.split(".")
            module_unet = (
                ".".join(name[:-3]) + ".to_" + name[-2].split("_")[1] + ".weight"
            )
            unet_state_dict[module_unet] = weight

    self.load_state_dict(unet_state_dict)

    warn_messages = "you are loading custom diffusion weights into a U-Net without the official attention processor, which may cause potential issues. Please ensure you understand the implications of your actions."
    warnings.warn(warn_messages)
