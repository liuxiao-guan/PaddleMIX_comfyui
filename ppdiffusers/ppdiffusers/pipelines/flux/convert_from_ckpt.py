# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the Stable Diffusion checkpoints."""

import re
from io import BytesIO
from typing import Dict, Optional, Union
from ppdiffusers.transformers import AutoConfig

import numpy as np
import paddle
import requests
from contextlib import nullcontext


from ppdiffusers.transformers import (
    BertTokenizer,
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    T5Tokenizer,
    T5EncoderModel,
)
import transformers

from ...models import (
    AutoencoderKL,
    ControlNetModel,
    PriorTransformer,
    UNet2DConditionModel,
    FluxTransformer2DModel,
)
from ...schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UnCLIPScheduler,
    FlowMatchEulerDiscreteScheduler,


)
from ...utils import is_accelerate_available, is_omegaconf_available, is_paddlenlp_available, logging, smart_load
from ...utils.import_utils import BACKENDS_MAPPING
from ..latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
from ..paint_by_example import PaintByExampleImageEncoder
from ..pipeline_utils import DiffusionPipeline
# from .safety_checker import StableDiffusionSafetyChecker
# from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

# if is_accelerate_available():
#     from accelerate import init_empty_weights
#     from ..models.modeling_utils import load_model_dict_into_meta
LDM_VAE_KEYS = ["first_stage_model.", "vae."]
DIFFUSERS_TO_LDM_MAPPING = {
    "unet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "conv_norm_out.weight": "out.0.weight",
            "conv_norm_out.bias": "out.0.bias",
            "conv_out.weight": "out.2.weight",
            "conv_out.bias": "out.2.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "controlnet": {
        "layers": {
            "time_embedding.linear_1.weight": "time_embed.0.weight",
            "time_embedding.linear_1.bias": "time_embed.0.bias",
            "time_embedding.linear_2.weight": "time_embed.2.weight",
            "time_embedding.linear_2.bias": "time_embed.2.bias",
            "conv_in.weight": "input_blocks.0.0.weight",
            "conv_in.bias": "input_blocks.0.0.bias",
            "controlnet_cond_embedding.conv_in.weight": "input_hint_block.0.weight",
            "controlnet_cond_embedding.conv_in.bias": "input_hint_block.0.bias",
            "controlnet_cond_embedding.conv_out.weight": "input_hint_block.14.weight",
            "controlnet_cond_embedding.conv_out.bias": "input_hint_block.14.bias",
        },
        "class_embed_type": {
            "class_embedding.linear_1.weight": "label_emb.0.0.weight",
            "class_embedding.linear_1.bias": "label_emb.0.0.bias",
            "class_embedding.linear_2.weight": "label_emb.0.2.weight",
            "class_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
        "addition_embed_type": {
            "add_embedding.linear_1.weight": "label_emb.0.0.weight",
            "add_embedding.linear_1.bias": "label_emb.0.0.bias",
            "add_embedding.linear_2.weight": "label_emb.0.2.weight",
            "add_embedding.linear_2.bias": "label_emb.0.2.bias",
        },
    },
    "vae": {
        "encoder.conv_in.weight": "encoder.conv_in.weight",
        "encoder.conv_in.bias": "encoder.conv_in.bias",
        "encoder.conv_out.weight": "encoder.conv_out.weight",
        "encoder.conv_out.bias": "encoder.conv_out.bias",
        "encoder.conv_norm_out.weight": "encoder.norm_out.weight",
        "encoder.conv_norm_out.bias": "encoder.norm_out.bias",
        "decoder.conv_in.weight": "decoder.conv_in.weight",
        "decoder.conv_in.bias": "decoder.conv_in.bias",
        "decoder.conv_out.weight": "decoder.conv_out.weight",
        "decoder.conv_out.bias": "decoder.conv_out.bias",
        "decoder.conv_norm_out.weight": "decoder.norm_out.weight",
        "decoder.conv_norm_out.bias": "decoder.norm_out.bias",
        "quant_conv.weight": "quant_conv.weight",
        "quant_conv.bias": "quant_conv.bias",
        "post_quant_conv.weight": "post_quant_conv.weight",
        "post_quant_conv.bias": "post_quant_conv.bias",
    },
    "openclip": {
        "layers": {
            "text_model.embeddings.position_embedding.weight": "positional_embedding",
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.final_layer_norm.weight": "ln_final.weight",
            "text_model.final_layer_norm.bias": "ln_final.bias",
            "text_projection.weight": "text_projection",
        },
        "transformer": {
            "text_model.encoder.layers.": "resblocks.",
            "layer_norm1": "ln_1",
            "layer_norm2": "ln_2",
            ".fc1.": ".c_fc.",
            ".fc2.": ".c_proj.",
            ".self_attn": ".attn",
            "transformer.text_model.final_layer_norm.": "ln_final.",
            "transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "transformer.text_model.embeddings.position_embedding.weight": "positional_embedding",
        },
    },
}
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from ...models.modeling_utils import ContextManagers, faster_set_state_dict

if is_paddlenlp_available():
    try:
        from paddlenlp.transformers.model_utils import no_init_weights
    except ImportError:
        from ...utils.paddle_utils import no_init_weights


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "to_q.weight")
        new_item = new_item.replace("q.bias", "to_q.bias")

        new_item = new_item.replace("k.weight", "to_k.weight")
        new_item = new_item.replace("k.bias", "to_k.bias")

        new_item = new_item.replace("v.weight", "to_v.weight")
        new_item = new_item.replace("v.bias", "to_v.bias")

        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = np.split(old_tensor, 3, axis=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        is_attn_weight = "proj_attn.weight" in new_path or ("attentions" in new_path and "to_" in new_path)
        shape = old_checkpoint[path["old"]].shape
        if is_attn_weight and len(shape) == 3:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        elif is_attn_weight and len(shape) == 4:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config, image_size: int, controlnet=False):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if controlnet:
        unet_params = original_config.model.params.control_stage_config.params
    else:
        if "unet_config" in original_config.model.params and original_config.model.params.unet_config is not None:
            unet_params = original_config.model.params.unet_config.params
        else:
            unet_params = original_config.model.params.network_config.params

    vae_params = original_config.model.params.first_stage_config.params.ddconfig

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    if unet_params.transformer_depth is not None:
        transformer_layers_per_block = (
            unet_params.transformer_depth
            if isinstance(unet_params.transformer_depth, int)
            else list(unet_params.transformer_depth)
        )
    else:
        transformer_layers_per_block = 1

    vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)

    head_dim = unet_params.num_heads if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params.use_linear_in_transformer if "use_linear_in_transformer" in unet_params else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim_mult = unet_params.model_channels // unet_params.num_head_channels
            head_dim = [head_dim_mult * c for c in list(unet_params.channel_mult)]

    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None

    if unet_params.context_dim is not None:
        context_dim = (
            unet_params.context_dim if isinstance(unet_params.context_dim, int) else unet_params.context_dim[0]
        )

    if "num_classes" in unet_params:
        if unet_params.num_classes == "sequential":
            if context_dim in [2048, 1280]:
                # SDXL
                addition_embed_type = "text_time"
                addition_time_embed_dim = 256
            else:
                class_embed_type = "projection"
            assert "adm_in_channels" in unet_params
            projection_class_embeddings_input_dim = unet_params.adm_in_channels

    config = {
        "sample_size": image_size // vae_scale_factor,
        "in_channels": unet_params.in_channels,
        "down_block_types": tuple(down_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": unet_params.num_res_blocks,
        "cross_attention_dim": context_dim,
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "addition_embed_type": addition_embed_type,
        "addition_time_embed_dim": addition_time_embed_dim,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "transformer_layers_per_block": transformer_layers_per_block,
    }

    if "disable_self_attentions" in unet_params:
        config["only_cross_attention"] = unet_params.disable_self_attentions

    if "num_classes" in unet_params and isinstance(unet_params.num_classes, int):
        config["num_class_embeds"] = unet_params.num_classes

    if controlnet:
        config["conditioning_channels"] = unet_params.hint_channels
    else:
        config["out_channels"] = unet_params.out_channels
        config["up_block_types"] = tuple(up_block_types)

    return config


def create_vae_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": image_size,
        "in_channels": vae_params.in_channels,
        "out_channels": vae_params.out_ch,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params.z_channels,
        "layers_per_block": vae_params.num_res_blocks,
    }
    return config

def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping["old"], mapping["new"]).replace("nin_shortcut", "conv_shortcut")
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)

def update_vae_attentions_ldm_to_diffusers(keys, new_checkpoint, checkpoint, mapping):
    for ldm_key in keys:
        diffusers_key = (
            ldm_key.replace(mapping["old"], mapping["new"])
            .replace("norm.weight", "group_norm.weight")
            .replace("norm.bias", "group_norm.bias")
            .replace("q.weight", "to_q.weight")
            .replace("q.bias", "to_q.bias")
            .replace("k.weight", "to_k.weight")
            .replace("k.bias", "to_k.bias")
            .replace("v.weight", "to_v.weight")
            .replace("v.bias", "to_v.bias")
            .replace("proj_out.weight", "to_out.0.weight")
            .replace("proj_out.bias", "to_out.0.bias")
        )
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)

        # proj_attn.weight has to be converted from conv 1D to linear
        shape = new_checkpoint[diffusers_key].shape

        if len(shape) == 3:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0]
        elif len(shape) == 4:
            new_checkpoint[diffusers_key] = new_checkpoint[diffusers_key][:, :, 0, 0]

def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    # remove the LDM_VAE_KEY prefix from the ldm checkpoint keys so that it is easier to map them to diffusers keys
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = ""
    for ldm_vae_key in LDM_VAE_KEYS:
        if any(k.startswith(ldm_vae_key) for k in keys):
            vae_key = ldm_vae_key

    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}
    vae_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING["vae"]
    for diffusers_key, ldm_key in vae_diffusers_ldm_map.items():
        if ldm_key not in vae_state_dict:
            continue
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(config["down_block_types"])
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"},
        )
        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.get(
                f"encoder.down.{i}.downsample.conv.bias"
            )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(config["up_block_types"])
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"},
        )
        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]
        update_vae_resnet_ldm_to_diffusers(
            resnets,
            new_checkpoint,
            vae_state_dict,
            mapping={"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"},
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    update_vae_attentions_ldm_to_diffusers(
        mid_attentions, new_checkpoint, vae_state_dict, mapping={"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    )
    conv_attn_to_linear(new_checkpoint)

    return new_checkpoint

def convert_flux_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    converted_state_dict = {}
    keys = list(checkpoint.keys())

    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    num_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "double_blocks." in k))[-1] + 1  # noqa: C401
    num_single_layers = list(set(int(k.split(".", 2)[1]) for k in checkpoint if "single_blocks." in k))[-1] + 1  # noqa: C401
    mlp_ratio = 4.0
    inner_dim = 3072

    # in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
    # while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
    def swap_scale_shift(weight):
        weight = paddle.to_tensor(weight)
        shift, scale = weight.chunk( chunks=2, axis=0)
        new_weight = paddle.concat([scale, shift], axis=0)
        return new_weight

    ## time_text_embed.timestep_embedder <-  time_in
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = checkpoint.pop(
        "time_in.in_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = checkpoint.pop("time_in.in_layer.bias")
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = checkpoint.pop(
        "time_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = checkpoint.pop("time_in.out_layer.bias")

    ## time_text_embed.text_embedder <- vector_in
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = checkpoint.pop("vector_in.in_layer.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = checkpoint.pop("vector_in.in_layer.bias")
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = checkpoint.pop(
        "vector_in.out_layer.weight"
    )
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = checkpoint.pop("vector_in.out_layer.bias")

    # guidance
    has_guidance = any("guidance" in k for k in checkpoint)
    if has_guidance:
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = checkpoint.pop(
            "guidance_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = checkpoint.pop(
            "guidance_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = checkpoint.pop(
            "guidance_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = checkpoint.pop(
            "guidance_in.out_layer.bias"
        )

    # context_embedder
    converted_state_dict["context_embedder.weight"] = checkpoint.pop("txt_in.weight")
    converted_state_dict["context_embedder.bias"] = checkpoint.pop("txt_in.bias")

    # x_embedder
    converted_state_dict["x_embedder.weight"] = checkpoint.pop("img_in.weight")
    converted_state_dict["x_embedder.bias"] = checkpoint.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms.
        ## norm1
        converted_state_dict[f"{block_prefix}norm1.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_mod.lin.bias"
        )
        ## norm1_context
        converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mod.lin.bias"
        )
        # Q, K, V
        sample_q, sample_k, sample_v = paddle.chunk(paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.weight")), chunks=3, axis=0)
        context_q, context_k, context_v = paddle.chunk(
            paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.weight")), chunks=3, axis=0
        )
        sample_q_bias, sample_k_bias, sample_v_bias = paddle.chunk(
            paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.img_attn.qkv.bias")), chunks=3, axis=0
        )
        context_q_bias, context_k_bias, context_v_bias = paddle.chunk(
            paddle.to_tensor(checkpoint.pop(f"double_blocks.{i}.txt_attn.qkv.bias")), chunks=3, axis=0
        )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = sample_q
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = sample_q_bias
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = sample_k
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = sample_k_bias
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = sample_v
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = sample_v_bias
        converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = context_q
        converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = context_q_bias
        converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = context_k
        converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = context_k_bias
        converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = context_v
        converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = context_v_bias
        # qk_norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.norm.key_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
        )
        # ff img_mlp
        converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.0.bias")
        converted_state_dict[f"{block_prefix}ff.net.2.weight"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.weight")
        converted_state_dict[f"{block_prefix}ff.net.2.bias"] = checkpoint.pop(f"double_blocks.{i}.img_mlp.2.bias")
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.0.bias"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.weight"
        )
        converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_mlp.2.bias"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = checkpoint.pop(
            f"double_blocks.{i}.img_attn.proj.bias"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.weight"
        )
        converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = checkpoint.pop(
            f"double_blocks.{i}.txt_attn.proj.bias"
        )

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear  <- single_blocks.0.modulation.lin
        converted_state_dict[f"{block_prefix}norm.linear.weight"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.weight"
        )
        converted_state_dict[f"{block_prefix}norm.linear.bias"] = checkpoint.pop(
            f"single_blocks.{i}.modulation.lin.bias"
        )
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        #q, k, v, mlp = torch.split(checkpoint.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
        q, k, v, mlp = paddle.split(paddle.to_tensor(checkpoint.pop(f"single_blocks.{i}.linear1.weight")), num_or_sections=split_size, axis=0)
        q_bias, k_bias, v_bias, mlp_bias = paddle.split(
            paddle.to_tensor(checkpoint.pop(f"single_blocks.{i}.linear1.bias")), num_or_sections=split_size, axis=0
        )
        # q_bias, k_bias, v_bias, mlp_bias = torch.split(
        #     checkpoint.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
        # )
        converted_state_dict[f"{block_prefix}attn.to_q.weight"] = q
        converted_state_dict[f"{block_prefix}attn.to_q.bias"] = q_bias
        converted_state_dict[f"{block_prefix}attn.to_k.weight"] = k
        converted_state_dict[f"{block_prefix}attn.to_k.bias"] = k_bias
        converted_state_dict[f"{block_prefix}attn.to_v.weight"] = v
        converted_state_dict[f"{block_prefix}attn.to_v.bias"] = v_bias
        converted_state_dict[f"{block_prefix}proj_mlp.weight"] = mlp
        converted_state_dict[f"{block_prefix}proj_mlp.bias"] = mlp_bias
        # qk norm
        converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.query_norm.scale"
        )
        converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = checkpoint.pop(
            f"single_blocks.{i}.norm.key_norm.scale"
        )
        # output projections.
        converted_state_dict[f"{block_prefix}proj_out.weight"] = checkpoint.pop(f"single_blocks.{i}.linear2.weight")
        converted_state_dict[f"{block_prefix}proj_out.bias"] = checkpoint.pop(f"single_blocks.{i}.linear2.bias")

    converted_state_dict["proj_out.weight"] = checkpoint.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = checkpoint.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.weight")
    )
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(
        checkpoint.pop("final_layer.adaLN_modulation.1.bias")
    )

    return converted_state_dict

def convert_open_clip_checkpoint(
    checkpoint,
    config_name,
    prefix="cond_stage_model.model.",
    has_projection=False,
    local_files_only=False,
    **config_kwargs,
):
    # text_model = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
    # text_model = CLIPTextModelWithProjection.from_pretrained(
    #    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", projection_dim=1280
    # )
    try:
        config = CLIPTextConfig.from_pretrained(config_name, **config_kwargs, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: '{config_name}'."
        )

    init_contexts = []
    init_contexts.append(paddle.dtype_guard(paddle.float32))
    init_contexts.append(no_init_weights(_enable=True))
    if hasattr(paddle, "LazyGuard"):
        init_contexts.append(paddle.LazyGuard())
    with ContextManagers(init_contexts):
        text_model = CLIPTextModelWithProjection(config) if has_projection else CLIPTextModel(config)

    keys = list(checkpoint.keys())

    keys_to_ignore = []
    if config_name == "stabilityai/stable-diffusion-2" and config.num_hidden_layers == 23:
        # make sure to remove all keys > 22
        keys_to_ignore += [k for k in keys if k.startswith("cond_stage_model.model.transformer.resblocks.23")]
        keys_to_ignore += ["cond_stage_model.model.text_projection"]

    text_model_dict = {}

    if prefix + "text_projection" in checkpoint:
        d_model = int(checkpoint[prefix + "text_projection"].shape[0])
    else:
        d_model = 1024

    # text_model_dict["text_model.embeddings.position_embedding.weight"] = text_model.text_model.embeddings.get_buffer("position_ids")

    for key in keys:
        if key in keys_to_ignore:
            continue

        if key[len(prefix) :] in textenc_conversion_map:
            if key.endswith("text_projection"):
                value = checkpoint[key].T
            else:
                value = checkpoint[key]

            text_model_dict[textenc_conversion_map[key[len(prefix) :]]] = value

        if key.startswith(prefix + "transformer."):
            new_key = key[len(prefix + "transformer.") :]
            if new_key.endswith(".in_proj_weight"):
                new_key = new_key[: -len(".in_proj_weight")]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][:d_model, :]
                text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][d_model : d_model * 2, :]
                text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][d_model * 2 :, :]
            elif new_key.endswith(".in_proj_bias"):
                new_key = new_key[: -len(".in_proj_bias")]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                text_model_dict[new_key + ".q_proj.bias"] = checkpoint[key][:d_model]
                text_model_dict[new_key + ".k_proj.bias"] = checkpoint[key][d_model : d_model * 2]
                text_model_dict[new_key + ".v_proj.bias"] = checkpoint[key][d_model * 2 :]
            else:
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)

                text_model_dict[new_key] = checkpoint[key]

    if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
        text_model_dict.pop("text_model.embeddings.position_ids", None)
    faster_set_state_dict(text_model, convert_diffusers_vae_unet_to_ppdiffusers(text_model, text_model_dict))

    return text_model
def get_default(params, key, default):
    if key in params:
        return params[key]
    else:
        return default


def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config.model.params.timesteps,
        beta_start=original_config.model.params.linear_start,
        beta_end=original_config.model.params.linear_end,
        beta_schedule="scaled_linear",
    )
    return schedular


def create_ldm_bert_config(original_config):
    bert_params = dict(original_config.model.params.cond_stage_config.params)
    config = dict(
        vocab_size=get_default(bert_params, "vocab_size", 30522),
        max_position_embeddings=get_default(bert_params, "max_seq_len", 77),
        encoder_layers=get_default(bert_params, "n_layer", 32),
        encoder_ffn_dim=get_default(bert_params, "n_embed", 1280) * 4,
        encoder_attention_heads=8,
        head_dim=64,
        activation_function="gelu",
        d_model=get_default(bert_params, "n_embed", 1280),
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        pad_token_id=0,
    )
    return LDMBertConfig(**config)


def convert_ldm_unet_checkpoint(
    checkpoint, config, path=None, extract_ema=False, controlnet=False, skip_extract_state_dict=False
):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    if skip_extract_state_dict:
        unet_state_dict = checkpoint
    else:
        # extract state_dict for UNet
        unet_state_dict = {}
        keys = list(checkpoint.keys())

        if controlnet:
            unet_key = "control_model."
        else:
            unet_key = "model.diffusion_model."

        # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
        if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
            logger.warning(f"Checkpoint {path} has both EMA and non-EMA weights.")
            logger.warning(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
        else:
            if sum(k.startswith("model_ema") for k in keys) > 100:
                logger.warning(
                    "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                    " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
                )

            for key in keys:
                if key.startswith(unet_key):
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    if config["class_embed_type"] is None:
        # No parameters to port
        ...
    elif config["class_embed_type"] == "timestep" or config["class_embed_type"] == "projection":
        new_checkpoint["class_embedding.linear_1.weight"] = unet_state_dict["label_emb.0.0.weight"]
        new_checkpoint["class_embedding.linear_1.bias"] = unet_state_dict["label_emb.0.0.bias"]
        new_checkpoint["class_embedding.linear_2.weight"] = unet_state_dict["label_emb.0.2.weight"]
        new_checkpoint["class_embedding.linear_2.bias"] = unet_state_dict["label_emb.0.2.bias"]
    else:
        raise NotImplementedError(f"Not implemented `class_embed_type`: {config['class_embed_type']}")

    if config["addition_embed_type"] == "text_time":
        new_checkpoint["add_embedding.linear_1.weight"] = unet_state_dict["label_emb.0.0.weight"]
        new_checkpoint["add_embedding.linear_1.bias"] = unet_state_dict["label_emb.0.0.bias"]
        new_checkpoint["add_embedding.linear_2.weight"] = unet_state_dict["label_emb.0.2.weight"]
        new_checkpoint["add_embedding.linear_2.bias"] = unet_state_dict["label_emb.0.2.bias"]

    # Relevant to StableDiffusionUpscalePipeline
    if "num_class_embeds" in config:
        new_checkpoint["class_embedding.weight"] = unet_state_dict["label_emb.weight"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    if not controlnet:
        new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
        new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
        new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
        new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)

            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    if controlnet:
        # conditioning embedding

        orig_index = 0

        new_checkpoint["controlnet_cond_embedding.conv_in.weight"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.weight"
        )
        new_checkpoint["controlnet_cond_embedding.conv_in.bias"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.bias"
        )

        orig_index += 2

        diffusers_index = 0

        while diffusers_index < 6:
            new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_index}.weight"] = unet_state_dict.pop(
                f"input_hint_block.{orig_index}.weight"
            )
            new_checkpoint[f"controlnet_cond_embedding.blocks.{diffusers_index}.bias"] = unet_state_dict.pop(
                f"input_hint_block.{orig_index}.bias"
            )
            diffusers_index += 1
            orig_index += 2

        new_checkpoint["controlnet_cond_embedding.conv_out.weight"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.weight"
        )
        new_checkpoint["controlnet_cond_embedding.conv_out.bias"] = unet_state_dict.pop(
            f"input_hint_block.{orig_index}.bias"
        )

        # down blocks
        for i in range(num_input_blocks):
            new_checkpoint[f"controlnet_down_blocks.{i}.weight"] = unet_state_dict.pop(f"zero_convs.{i}.0.weight")
            new_checkpoint[f"controlnet_down_blocks.{i}.bias"] = unet_state_dict.pop(f"zero_convs.{i}.0.bias")

        # mid block
        new_checkpoint["controlnet_mid_block.weight"] = unet_state_dict.pop("middle_block_out.0.weight")
        new_checkpoint["controlnet_mid_block.bias"] = unet_state_dict.pop("middle_block_out.0.bias")

    return new_checkpoint


def convert_diffusers_vae_unet_to_ppdiffusers(vae_or_unet, diffusers_vae_unet_checkpoint):
    import paddle.nn as nn

    need_transpose = []
    for k, v in vae_or_unet.named_sublayers(include_self=True):
        if isinstance(v, nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = {}
    for k in list(diffusers_vae_unet_checkpoint.keys()):
        v = diffusers_vae_unet_checkpoint.pop(k)
        if k not in need_transpose:
            new_vae_or_unet[k] = v
        else:
            new_vae_or_unet[k] = v.T
    return new_vae_or_unet


def convert_ldm_bert_checkpoint(checkpoint, config):
    # extract state dict for bert
    bert_state_dict = {}
    bert_key = "cond_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(bert_key):
            bert_state_dict[key.replace(bert_key, "")] = checkpoint.get(key)

    new_checkpoint = {}
    new_checkpoint["model.embeddings.embed_tokens.weight"] = bert_state_dict["transformer.token_emb.weight"]
    new_checkpoint["model.embeddings.embed_positions.weight"] = bert_state_dict["transformer.pos_emb.emb.weight"]
    for i in range(config.encoder_layers):
        double_i = 2 * i
        double_i_plus1 = 2 * i + 1
        # convert norm
        new_checkpoint[f"model.layers.{i}.self_attn_layer_norm.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i}.0.weight"
        ]
        new_checkpoint[f"model.layers.{i}.self_attn_layer_norm.bias"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i}.0.bias"
        ]

        new_checkpoint[f"model.layers.{i}.self_attn.q_proj.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i}.1.to_q.weight"
        ].T
        new_checkpoint[f"model.layers.{i}.self_attn.k_proj.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i}.1.to_k.weight"
        ].T
        new_checkpoint[f"model.layers.{i}.self_attn.v_proj.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i}.1.to_v.weight"
        ].T
        new_checkpoint[f"model.layers.{i}.self_attn.out_proj.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i}.1.to_out.weight"
        ].T
        new_checkpoint[f"model.layers.{i}.self_attn.out_proj.bias"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i}.1.to_out.bias"
        ]

        new_checkpoint[f"model.layers.{i}.final_layer_norm.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i_plus1}.0.weight"
        ]
        new_checkpoint[f"model.layers.{i}.final_layer_norm.bias"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i_plus1}.0.bias"
        ]
        new_checkpoint[f"model.layers.{i}.fc1.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.weight"
        ].T
        new_checkpoint[f"model.layers.{i}.fc1.bias"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.bias"
        ]
        new_checkpoint[f"model.layers.{i}.fc2.weight"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i_plus1}.1.net.2.weight"
        ].T
        new_checkpoint[f"model.layers.{i}.fc2.bias"] = bert_state_dict[
            f"transformer.attn_layers.layers.{double_i_plus1}.1.net.2.bias"
        ].T

    new_checkpoint["model.layer_norm.weight"] = bert_state_dict["transformer.norm.weight"]
    new_checkpoint["model.layer_norm.bias"] = bert_state_dict["transformer.norm.bias"]
    ldmbert = LDMBertModel(config)
    ldmbert.eval()
    faster_set_state_dict(ldmbert, new_checkpoint)
    return ldmbert


def convert_ldm_clip_checkpoint(checkpoint, local_files_only=False, text_encoder=None):
    if text_encoder is None:
        config_name = "openai/clip-vit-large-patch14"
        try:
            config = CLIPTextConfig.from_pretrained(config_name, local_files_only=local_files_only)
        except Exception:
            raise ValueError(
                f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: 'openai/clip-vit-large-patch14'."
            )
        init_contexts = []
        init_contexts.append(paddle.dtype_guard(paddle.float32))
        init_contexts.append(no_init_weights(_enable=True))
        if hasattr(paddle, "LazyGuard"):
            init_contexts.append(paddle.LazyGuard())
        with ContextManagers(init_contexts):
            text_model = CLIPTextModel(config)
    else:
        text_model = text_encoder

    keys = list(checkpoint.keys())

    text_model_dict = {}

    remove_prefixes = ["cond_stage_model.transformer", "conditioner.embedders.0.transformer","text_encoders.clip_l.transformer"]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                text_model_dict[key[len(prefix + ".") :]] = checkpoint[key]

    if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
        text_model_dict.pop("text_model.embeddings.position_ids", None)

    faster_set_state_dict(text_model, convert_diffusers_vae_unet_to_ppdiffusers(text_model, text_model_dict))

    return text_model
def convert_sd3_t5_checkpoint_to_diffusers(checkpoint):  
    keys = list(checkpoint.keys()) 
    text_model_dict = {}

    remove_prefixes = ["text_encoders.t5xxl.transformer."]
    
    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, "")
                text_model_dict[diffusers_key] = checkpoint.get(key)

    return text_model_dict


def create_diffusers_t5_model_from_checkpoint(
    cls,
    checkpoint,
    subfolder="",
    config=None,
    torch_dtype=None,
    local_files_only=None,
):
    if config:
        config = {"pretrained_model_name_or_path": config}
    # else:
    #     config = fetch_diffusers_config(checkpoint)
    # config = {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev" }
    # subfolder =  "text_encoder_2"
    # model_config = cls.config_class.from_pretrained(**config, subfolder=subfolder, local_files_only=local_files_only)
    # ctx = init_empty_weights if is_accelerate_available() else nullcontext
    # with ctx():
    #     model = cls(model_config)
    config = AutoConfig.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2")
    init_contexts = []
    init_contexts.append(paddle.dtype_guard(paddle.float32))
    init_contexts.append(no_init_weights(_enable=True))
    if hasattr(paddle, "LazyGuard"):
        init_contexts.append(paddle.LazyGuard())
    with ContextManagers(init_contexts):
        model = T5EncoderModel(config)

    diffusers_format_checkpoint = convert_sd3_t5_checkpoint_to_diffusers(checkpoint)

    # if is_accelerate_available():
    #     load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
    # else:
    #model.load_state_dict(diffusers_format_checkpoint)
    faster_set_state_dict(model, convert_diffusers_vae_unet_to_ppdiffusers(model, diffusers_format_checkpoint))

    use_keep_in_fp32_modules = (T5EncoderModel._keep_in_fp32_modules is not None) and (torch_dtype == paddle.float16)
    if use_keep_in_fp32_modules:
        keep_in_fp32_modules = model._keep_in_fp32_modules
    else:
        keep_in_fp32_modules = []

    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                # param = param.to(torch.float32) does not work here as only in the local scope.
                param.data = param.data.to(paddle.float32)
    

    return model

textenc_conversion_lst = [
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("ln_final.weight", "text_model.final_layer_norm.weight"),
    ("ln_final.bias", "text_model.final_layer_norm.bias"),
    ("text_projection", "text_projection.weight"),
]
textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

textenc_transformer_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    ("token_embedding.weight", "transformer.text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "transformer.text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))


def convert_paint_by_example_checkpoint(checkpoint, local_files_only=False):
    config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14", local_files_only=local_files_only)
    model = PaintByExampleImageEncoder(config)

    keys = list(checkpoint.keys())

    text_model_dict = {}

    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[key]

    # load clip vision
    faster_set_state_dict(model.model, convert_diffusers_vae_unet_to_ppdiffusers(model.model, text_model_dict))

    # load mapper
    keys_mapper = {
        k[len("cond_stage_model.mapper.res") :]: v
        for k, v in checkpoint.items()
        if k.startswith("cond_stage_model.mapper")
    }

    MAPPING = {
        "attn.c_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
        "attn.c_proj": ["attn1.to_out.0"],
        "ln_1": ["norm1"],
        "ln_2": ["norm3"],
        "mlp.c_fc": ["ff.net.0.proj"],
        "mlp.c_proj": ["ff.net.2"],
    }

    mapped_weights = {}
    for key, value in keys_mapper.items():
        prefix = key[: len("blocks.i")]
        suffix = key.split(prefix)[-1].split(".")[-1]
        name = key.split(prefix)[-1].split(suffix)[0][1:-1]
        mapped_names = MAPPING[name]

        num_splits = len(mapped_names)
        for i, mapped_name in enumerate(mapped_names):
            new_name = ".".join([prefix, mapped_name, suffix])
            shape = value.shape[0] // num_splits
            mapped_weights[new_name] = value[i * shape : (i + 1) * shape]

    faster_set_state_dict(model.mapper, convert_diffusers_vae_unet_to_ppdiffusers(model.mapper, mapped_weights))

    # load final layer norm
    faster_set_state_dict(
        model.final_layer_norm,
        {
            "bias": checkpoint["cond_stage_model.final_ln.bias"],
            "weight": checkpoint["cond_stage_model.final_ln.weight"],
        },
    )

    # load final proj
    faster_set_state_dict(
        model.proj_out,
        {
            "bias": checkpoint["proj_out.bias"],
            "weight": checkpoint["proj_out.weight"].T,
        },
    )

    # load uncond vector
    model.uncond_vector.set_value(checkpoint["learnable_vector"])
    return model


def convert_open_clip_checkpoint(
    checkpoint,
    config_name,
    prefix="cond_stage_model.model.",
    has_projection=False,
    local_files_only=False,
    **config_kwargs,
):
    # text_model = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
    # text_model = CLIPTextModelWithProjection.from_pretrained(
    #    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", projection_dim=1280
    # )
    try:
        config = CLIPTextConfig.from_pretrained(config_name, **config_kwargs, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: '{config_name}'."
        )

    init_contexts = []
    init_contexts.append(paddle.dtype_guard(paddle.float32))
    init_contexts.append(no_init_weights(_enable=True))
    if hasattr(paddle, "LazyGuard"):
        init_contexts.append(paddle.LazyGuard())
    with ContextManagers(init_contexts):
        text_model = CLIPTextModelWithProjection(config) if has_projection else CLIPTextModel(config)

    keys = list(checkpoint.keys())

    keys_to_ignore = []
    if config_name == "stabilityai/stable-diffusion-2" and config.num_hidden_layers == 23:
        # make sure to remove all keys > 22
        keys_to_ignore += [k for k in keys if k.startswith("cond_stage_model.model.transformer.resblocks.23")]
        keys_to_ignore += ["cond_stage_model.model.text_projection"]

    text_model_dict = {}

    if prefix + "text_projection" in checkpoint:
        d_model = int(checkpoint[prefix + "text_projection"].shape[0])
    else:
        d_model = 1024

    # text_model_dict["text_model.embeddings.position_embedding.weight"] = text_model.text_model.embeddings.get_buffer("position_ids")

    for key in keys:
        if key in keys_to_ignore:
            continue

        if key[len(prefix) :] in textenc_conversion_map:
            if key.endswith("text_projection"):
                value = checkpoint[key].T
            else:
                value = checkpoint[key]

            text_model_dict[textenc_conversion_map[key[len(prefix) :]]] = value

        if key.startswith(prefix + "transformer."):
            new_key = key[len(prefix + "transformer.") :]
            if new_key.endswith(".in_proj_weight"):
                new_key = new_key[: -len(".in_proj_weight")]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][:d_model, :]
                text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][d_model : d_model * 2, :]
                text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][d_model * 2 :, :]
            elif new_key.endswith(".in_proj_bias"):
                new_key = new_key[: -len(".in_proj_bias")]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)
                text_model_dict[new_key + ".q_proj.bias"] = checkpoint[key][:d_model]
                text_model_dict[new_key + ".k_proj.bias"] = checkpoint[key][d_model : d_model * 2]
                text_model_dict[new_key + ".v_proj.bias"] = checkpoint[key][d_model * 2 :]
            else:
                new_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], new_key)

                text_model_dict[new_key] = checkpoint[key]

    if not (hasattr(text_model, "embeddings") and hasattr(text_model.embeddings.position_ids)):
        text_model_dict.pop("text_model.embeddings.position_ids", None)
    faster_set_state_dict(text_model, convert_diffusers_vae_unet_to_ppdiffusers(text_model, text_model_dict))

    return text_model


def stable_unclip_image_encoder(original_config, local_files_only=False):
    """
    Returns the image processor and clip image encoder for the img2img unclip pipeline.

    We currently know of two types of stable unclip models which separately use the clip and the openclip image
    encoders.
    """

    image_embedder_config = original_config.model.params.embedder_config

    sd_clip_image_embedder_class = image_embedder_config.target
    sd_clip_image_embedder_class = sd_clip_image_embedder_class.split(".")[-1]

    if sd_clip_image_embedder_class == "ClipImageEmbedder":
        clip_model_name = image_embedder_config.params.model

        if clip_model_name == "ViT-L/14":
            feature_extractor = CLIPImageProcessor()
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", local_files_only=local_files_only
            )
        else:
            raise NotImplementedError(f"Unknown CLIP checkpoint name in stable diffusion checkpoint {clip_model_name}")

    elif sd_clip_image_embedder_class == "FrozenOpenCLIPImageEmbedder":
        feature_extractor = CLIPImageProcessor()
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_files_only=local_files_only
        )
    else:
        raise NotImplementedError(
            f"Unknown CLIP image embedder class in stable diffusion checkpoint {sd_clip_image_embedder_class}"
        )

    return feature_extractor, image_encoder


def stable_unclip_image_noising_components(
    original_config, clip_stats_path: Optional[str] = None, device: Optional[str] = None
):
    """
    Returns the noising components for the img2img and txt2img unclip pipelines.

    Converts the stability noise augmentor into
    1. a `StableUnCLIPImageNormalizer` for holding the CLIP stats
    2. a `DDPMScheduler` for holding the noise schedule

    If the noise augmentor config specifies a clip stats path, the `clip_stats_path` must be provided.
    """
    noise_aug_config = original_config.model.params.noise_aug_config
    noise_aug_class = noise_aug_config.target
    noise_aug_class = noise_aug_class.split(".")[-1]

    if noise_aug_class == "CLIPEmbeddingNoiseAugmentation":
        noise_aug_config = noise_aug_config.params
        embedding_dim = noise_aug_config.timestep_dim
        max_noise_level = noise_aug_config.noise_schedule_config.timesteps
        beta_schedule = noise_aug_config.noise_schedule_config.beta_schedule

        image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=embedding_dim)
        image_noising_scheduler = DDPMScheduler(num_train_timesteps=max_noise_level, beta_schedule=beta_schedule)

        if "clip_stats_path" in noise_aug_config:
            if clip_stats_path is None:
                raise ValueError("This stable unclip config requires a `clip_stats_path`")

            from ...utils import torch_load

            clip_mean, clip_std = torch_load(clip_stats_path)
            if hasattr(clip_mean, "numpy"):
                clip_mean = clip_mean.numpy()
            if hasattr(clip_std, "numpy"):
                clip_std = clip_std.numpy()
            clip_mean = clip_mean[None, :]
            clip_std = clip_std[None, :]

            clip_stats_state_dict = {
                "mean": clip_mean,
                "std": clip_std,
            }
            faster_set_state_dict(image_normalizer, clip_stats_state_dict)
    else:
        raise NotImplementedError(f"Unknown noise augmentor class: {noise_aug_class}")

    return image_normalizer, image_noising_scheduler


def convert_controlnet_checkpoint(
    checkpoint,
    original_config,
    checkpoint_path,
    image_size,
    upcast_attention,
    extract_ema,
    use_linear_projection=None,
    cross_attention_dim=None,
):
    ctrlnet_config = create_unet_diffusers_config(original_config, image_size=image_size, controlnet=True)
    ctrlnet_config["upcast_attention"] = upcast_attention

    ctrlnet_config.pop("sample_size")

    if use_linear_projection is not None:
        ctrlnet_config["use_linear_projection"] = use_linear_projection

    if cross_attention_dim is not None:
        ctrlnet_config["cross_attention_dim"] = cross_attention_dim

    init_contexts = []
    init_contexts.append(paddle.dtype_guard(paddle.float32))
    init_contexts.append(no_init_weights(_enable=True))
    if hasattr(paddle, "LazyGuard"):
        init_contexts.append(paddle.LazyGuard())
    with ContextManagers(init_contexts):
        controlnet = ControlNetModel(**ctrlnet_config)

    # Some controlnet ckpt files are distributed independently from the rest of the
    # model components i.e. https://huggingface.co/thibaud/controlnet-sd21/
    if "time_embed.0.weight" in checkpoint:
        skip_extract_state_dict = True
    else:
        skip_extract_state_dict = False

    converted_ctrl_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint,
        ctrlnet_config,
        path=checkpoint_path,
        extract_ema=extract_ema,
        controlnet=True,
        skip_extract_state_dict=skip_extract_state_dict,
    )

    faster_set_state_dict(controlnet, convert_diffusers_vae_unet_to_ppdiffusers(controlnet, converted_ctrl_checkpoint))

    return controlnet


def download_from_original_flux_ckpt(
    checkpoint_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
    original_config_file: str = None,
    image_size: Optional[int] = None,
    prediction_type: str = None,
    model_type: str = None,
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: Optional[int] = None,
    upcast_attention: Optional[bool] = None,
    device: str = None,
    from_safetensors: bool = False,
    stable_unclip: Optional[str] = None,
    stable_unclip_prior: Optional[str] = None,
    clip_stats_path: Optional[str] = None,
    controlnet: Optional[bool] = None,
    adapter: Optional[bool] = None,
    load_safety_checker: bool = True,
    pipeline_class: DiffusionPipeline = None,
    local_files_only=False,
    vae_path=None,
    vae=None,
    text_encoder=None,
    tokenizer=None,
    config_files=None,
    paddle_dtype=None,
    **kwargs,
) -> DiffusionPipeline:
    """
    Load a Stable Diffusion pipeline object from a CompVis-style `.ckpt`/`.safetensors` file and (ideally) a `.yaml`
    config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path_or_dict (`str` or `dict`): Path to `.ckpt` file, or the state dict.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            inferred by looking for a key that only exists in SD2.0 models.
        image_size (`int`, *optional*, defaults to 512):
            The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Diffusion v2
            Base. Use 768 for Stable Diffusion v2.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. Use `'epsilon'` for Stable Diffusion v1.X and Stable
            Diffusion v2 Base. Use `'v_prediction'` for Stable Diffusion v2.
        num_in_channels (`int`, *optional*, defaults to None):
            The number of input channels. If `None`, it will be automatically inferred.
        scheduler_type (`str`, *optional*, defaults to 'pndm'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        model_type (`str`, *optional*, defaults to `None`):
            The pipeline type. `None` to automatically infer, or one of `["FrozenOpenCLIPEmbedder",
            "FrozenCLIPEmbedder", "PaintByExample"]`.
        is_img2img (`bool`, *optional*, defaults to `False`):
            Whether the model should be loaded as an img2img pipeline.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        upcast_attention (`bool`, *optional*, defaults to `None`):
            Whether the attention computation should always be upcasted. This is necessary when running stable
            diffusion 2.1.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of Paddle.
        load_safety_checker (`bool`, *optional*, defaults to `True`):
            Whether to load the safety checker or not. Defaults to `True`.
        pipeline_class (`str`, *optional*, defaults to `None`):
            The pipeline class to use. Pass `None` to determine automatically.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        vae (`AutoencoderKL`, *optional*, defaults to `None`):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. If
            this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
        text_encoder (`CLIPTextModel`, *optional*, defaults to `None`):
            An instance of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel)
            to use, specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
            variant. If this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
        tokenizer (`CLIPTokenizer`, *optional*, defaults to `None`):
            An instance of
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
            to use. If this parameter is `None`, the function will load a new instance of [CLIPTokenizer] by itself, if
            needed.
        config_files (`Dict[str, str]`, *optional*, defaults to `None`):
            A dictionary mapping from config file names to their contents. If this parameter is `None`, the function
            will load the config files by itself, if needed. Valid keys are:
                - `v1`: Config file for Stable Diffusion v1
                - `v2`: Config file for Stable Diffusion v2
                - `xl`: Config file for Stable Diffusion XL
                - `xl_refiner`: Config file for Stable Diffusion XL Refiner
        return: A StableDiffusionPipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """

    # import pipelines here to avoid circular import error when using from_single_file method
    from ppdiffusers import (
        LDMTextToImagePipeline,
        PaintByExamplePipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
        StableUnCLIPImg2ImgPipeline,
        StableUnCLIPPipeline,
        FluxPipeline,
    )


    if not is_omegaconf_available():
        raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

    from omegaconf import OmegaConf

    if isinstance(checkpoint_path_or_dict, str):
        checkpoint = smart_load(checkpoint_path_or_dict, return_numpy=True, return_global_step=True)

    elif isinstance(checkpoint_path_or_dict, dict):
        checkpoint = checkpoint_path_or_dict

    # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
    # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    global_step = int(checkpoint.pop("global_step", -1))

    if global_step == -1:
        print("global_step key not found in model")

    # must cast them to float32
    newcheckpoint = {}
    for k, v in checkpoint.items():
        try:
            if "int" in str(v.dtype):
                continue
        except Exception:
            continue
        newcheckpoint[k] = v.astype("float32")
    checkpoint = newcheckpoint
    

   
    # Convert the text model.
    # 此处先针对文生图的FLUX 没有针对其余功能的FLUX
    if ( model_type is None )and (pipeline_class == FluxPipeline):
        model_type = "Flux"
        if image_size is None:
            image_size = 1024
    
    num_train_timesteps = 1000

    if model_type in ["Flux"]:
        scheduler_dict = {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "num_train_timesteps": 1000,
            "shift": 3.0,
            "use_dynamic_shifting": True,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_dict)
        scheduler_type = "flowmatch-euler"
    else:
       pass
    
    converted_flux_checkpoint = convert_flux_transformer_checkpoint_to_diffusers(checkpoint)
    
    init_contexts = []
    init_contexts.append(paddle.dtype_guard(paddle.float32))
    init_contexts.append(no_init_weights(_enable=True))
    if hasattr(paddle, "LazyGuard"):
        init_contexts.append(paddle.LazyGuard())
    with ContextManagers(init_contexts):
        # 没有用配置文件 直接看官方文档的配置 与实例初始化的配置 是否一致 
        flux_transformer = FluxTransformer2DModel(guidance_embeds=True)
        

    faster_set_state_dict(flux_transformer, convert_diffusers_vae_unet_to_ppdiffusers(flux_transformer, converted_flux_checkpoint))

    
    # Convert the VAE model.
    if vae_path is None and vae is None:
        vae_config = AutoencoderKL.load_config("black-forest-labs/FLUX.1-dev", subfolder="vae")
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
        vae_path 
        init_contexts = []
        init_contexts.append(paddle.dtype_guard(paddle.float32))
        init_contexts.append(no_init_weights(_enable=True))
        if hasattr(paddle, "LazyGuard"):
            init_contexts.append(paddle.LazyGuard())
        with ContextManagers(init_contexts):
            vae = AutoencoderKL(**vae_config)

        faster_set_state_dict(vae, convert_diffusers_vae_unet_to_ppdiffusers(vae, converted_vae_checkpoint))

    elif vae is None:
        vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=local_files_only)

    
    if model_type in ["Flux"]:
        if model_type == "Flux":
            try:
                tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14", local_files_only=local_files_only
                )
            except Exception:
                raise ValueError(
                    f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'openai/clip-vit-large-patch14'."
                )
            text_encoder = convert_ldm_clip_checkpoint(checkpoint, local_files_only=local_files_only)
           
            

           

            # try:
            tokenizer_2 = T5Tokenizer.from_pretrained(
                "google/t5-v1_1-xxl", local_files_only=local_files_only
            )
            # except Exception:
            #     raise ValueError(
            #         f"With local_files_only set to {local_files_only}, you must first locally save the tokenizer in the following path: 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' with `pad_token` set to '!'."
            #     )

            config_name = "google/t5-v1_1-xxl"
            config_kwargs = {"projection_dim": 1280}
            text_encoder_2 = create_diffusers_t5_model_from_checkpoint(
                pipeline_class,
                checkpoint,
                subfolder="",
                config=None,
                torch_dtype=None,
                local_files_only=local_files_only,
            )
            #text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev",subfolder="text_encoder_2" , paddle_dtype="float32" )

            
            pipe = pipeline_class(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                transformer=flux_transformer,
                scheduler=scheduler,
                # force_zeros_for_empty_prompt=True,
            )
        
    else:
        text_config = create_ldm_bert_config(original_config)
        text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=local_files_only, model_max_length=77
        )
        pipe = LDMTextToImagePipeline(vqvae=vae, bert=text_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
    if paddle_dtype is not None:
        pipe.to(paddle_dtype=paddle_dtype)

    return pipe


def download_controlnet_from_original_ckpt(
    checkpoint_path: str,
    original_config_file: str,
    image_size: int = 512,
    extract_ema: bool = False,
    num_in_channels: Optional[int] = None,
    upcast_attention: Optional[bool] = None,
    device: str = None,
    from_safetensors: bool = False,
    use_linear_projection: Optional[bool] = None,
    cross_attention_dim: Optional[bool] = None,
) -> DiffusionPipeline:
    if not is_omegaconf_available():
        raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

    from omegaconf import OmegaConf

    checkpoint = smart_load(checkpoint_path, return_numpy=True)

    # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
    # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # must cast them to float32
    newcheckpoint = {}
    for k, v in checkpoint.items():
        try:
            if "int" in str(v.dtype):
                continue
        except Exception:
            continue
        newcheckpoint[k] = v.astype("float32")
    checkpoint = newcheckpoint

    original_config = OmegaConf.load(original_config_file)

    if num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

    if "control_stage_config" not in original_config.model.params:
        raise ValueError("`control_stage_config` not present in original config")

    controlnet = convert_controlnet_checkpoint(
        checkpoint,
        original_config,
        checkpoint_path,
        image_size,
        upcast_attention,
        extract_ema,
        use_linear_projection=use_linear_projection,
        cross_attention_dim=cross_attention_dim,
    )

    return controlnet
