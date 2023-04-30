import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Any, Union
from einops import rearrange


def compute_padded_dims(height, width, pad_multiple_of):
    padded_width = (int(math.ceil(width / pad_multiple_of)) * pad_multiple_of)
    padded_height = (int(math.ceil(height / pad_multiple_of)) * pad_multiple_of)
    return padded_height, padded_width


def pad_tensor_list(x: List[Tensor], padded_height: int, padded_width: int, stack: bool, pad_value: int) -> Union[Tensor, List[Tensor]]:
    for ii in range(len(x)):
        pad_right = padded_width - x[ii].shape[-1]
        pad_bottom = padded_height - x[ii].shape[-2]

        if x[ii].ndim == 2:
            x[ii] = F.pad(x[ii][None, None], (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)[0, 0]
        elif x[ii].ndim == 3:
            x[ii] = F.pad(x[ii][None], (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)[0]
        elif x[ii].ndim == 4:
            x[ii] = F.pad(x[ii], (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)
        elif x[ii].ndim == 5:
            d1, d2 = x[ii].shape[:2]
            x[ii] = F.pad(x[ii].flatten(0, 1), (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)
            x[ii] = x[ii].view(d1, d2, *x[ii].shape[1:])
        else:
            raise NotImplementedError(f"No implementation for ndims = {x[ii].ndim}")

    if stack:
        return torch.stack(x, 0)
    else:
        return x


def collate_fn_train(samples: List[Dict[str, Any]]):
    """
    Collate function for training data loader
    :param samples: list of samples
    :return:
    """
    # ensure that all samples come from the same dataset
    assert len(set([s['dataset'] for s in samples])) == 1, f"All batch samples must come from the same dataset"
    assert len(set([s['task_type'] for s in samples])) == 1, f"All batch samples must have the same task type"

    parsed_batch = {
        "images": [s["images"] for s in samples],
        "instance_masks": [s["instance_masks"] for s in samples],
    }

    if samples[0]["semantic_masks"] is None:
        parsed_batch["semantic_masks"] = None
        parsed_batch["ignore_masks"] = None
    else:  # panoptic seg sample
        parsed_batch["semantic_masks"] = [s["semantic_masks"] for s in samples]
        parsed_batch["ignore_masks"] = [s["ignore_masks"] for s in samples]
        # parsed_batch["thing_class_ids"] = samples[0]["thing_class_ids"]

    def combine_others(key):
        return [s[key] for s in samples]

    parsed_batch.update({
        key: combine_others(key)
        for key in ("meta", "class_ids", "dataset", "ref_frame_index")
        if key in samples[0]
    })

    parsed_batch["task_type"] = samples[0]["task_type"]

    return parsed_batch


# def collate_fn_inference(pad_multiple_of: int, samples: List[Dict[str, Any]]):
def collate_fn_inference(samples: List[Dict[str, Any]]):
    """
    Collate function for inference data loader
    :param pad_multiple_of:
    :param samples: list of samples
    :return:
    """
    assert len(samples) == 1, f"Batch size should be 1 during inference"

    images = samples[0]["images"]
    images = images.unsqueeze(0)  # [1, T, C, H, W]

    outputs = {
        "images": images,
        "frame_indices": samples[0]["frame_indices"],
        "task_type": samples[0]["task_type"],
        "orig_image_size": samples[0]["orig_image_size"],
    }

    if samples[0]["vos_ref_mask_info"] is not None:
        outputs["vos_ref_mask_info"] = samples[0]["vos_ref_mask_info"]
    else:
        outputs["vos_ref_mask_info"] = None

    return outputs


def collate_fn_coco_inference(samples: List[Dict[str, Any]]):
    """
    Collate function for inference data loader
    :param samples: list of samples
    :return:
    """
    assert len(samples) == 1, f"Batch size should be 1 during inference"
    height, width = samples[0]['images'].shape[-2:]
    padded_height, padded_width = compute_padded_dims(height, width)
    pad_bottom = padded_height - height
    pad_right = padded_width - width

    images = F.pad(samples[0]["images"], (0, pad_right, 0, pad_bottom), mode='constant', value=128)
    # images = images.unsqueeze(0)  # [1, T, C, H, W]

    outputs = {
        "images": images,
        "padding": (pad_right, pad_bottom),
    }

    outputs.update({
        k: v for k, v in samples[0].items() if k not in ("images", "padding", "frame_indices")
    })

    return outputs


def collate_fn_generic_image_sequence(samples: List[Dict[str, Any]]):
    max_height = max([s['images'].shape[-2] for s in samples])
    max_width = max([s['images'].shape[-1] for s in samples])
    padded_height, padded_width = compute_padded_dims(max_height, max_width)

    return {
        "images": pad_tensor_list([s["images"] for s in samples], padded_height, padded_width)
    }
