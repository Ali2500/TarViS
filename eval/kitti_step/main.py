"""
Expected result format:
- one folder per sequence with png files for every video frame.
- Each png file should contain a 3 channel RGB image of uint8 type with the following channel spec:
    - R: semantic ID
    - G: instance_id // 256
    - B: instance_id % 256
"""
from argparse import ArgumentParser
from eval.kitti_step.video_panoptic import STQuality
from tqdm import tqdm

import cv2
import numpy as np
import os
import os.path as osp
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable annoying warnings from tensorflow
import tensorflow as tf


VAL_SEQUENCE_NAMES = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
MAX_INSTANCES_PER_CATEGORY = 1000
NUM_CLASSES = 19


def rgb_to_eval_format(rgb_mask: np.ndarray) -> np.ndarray:
    # BGR images from openCV
    rgb_mask = rgb_mask.astype(np.int64)
    b, g, r = [rgb_mask[:, :, i] for i in range(3)]
    instance_map = (g * 256) + b
    semantic_map = r

    return (semantic_map * MAX_INSTANCES_PER_CATEGORY) + instance_map


def main(args):
    stq_metric = STQuality(num_classes=NUM_CLASSES,
                           things_list=[11, 13],
                           ignore_label=255,
                           max_instances_per_category=MAX_INSTANCES_PER_CATEGORY,
                           offset=int(1e6))

    pbar = tqdm(VAL_SEQUENCE_NAMES, leave=False)
    for seq_id, seq_name in enumerate(pbar):
        pbar.set_description(seq_name)
        pred_dir = osp.join(args.pred, seq_name)
        gt_dir = osp.join(args.gt, seq_name)

        assert osp.exists(pred_dir) and osp.exists(gt_dir), f"One of the following not found: {pred_dir} or {gt_dir}"

        filenames = sorted([x for x in os.listdir(gt_dir) if x.endswith(".png")])

        for fname in tqdm(filenames, leave=False):
            pred_png_path = osp.join(pred_dir, fname)
            assert osp.exists(pred_png_path), f"Image file not found: {pred_png_path}"

            pred_mask = cv2.imread(pred_png_path, cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(osp.join(gt_dir, fname), cv2.IMREAD_COLOR)

            # API expected single-channel mask arrays with the pixel value being:
            pred_mask = rgb_to_eval_format(pred_mask)
            gt_mask = rgb_to_eval_format(gt_mask)

            # convert to TF tensors
            pred_mask = tf.convert_to_tensor(pred_mask, tf.int64)
            gt_mask = tf.convert_to_tensor(gt_mask, tf.int64)

            stq_metric.update_state(gt_mask, pred_mask, sequence_id=seq_id)

    result = stq_metric.result()
    for k, v in result.items():
        print(f"{k}: {v}")

    def np_to_native_type(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif hasattr(x, "item"):
            return x.item()
        else:
            return x

    result = {k: np_to_native_type(v) for k, v in result.items()}
    result["IoU_per_seq"] = [x.item() for x in result["IoU_per_seq"]]

    with open(osp.join(osp.dirname(args.pred), "metrics.yaml"), 'w') as fh:
        yaml.dump(result, fh)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)

    main(parser.parse_args())
