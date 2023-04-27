# TarViS: A Unified Architecture for Target-based Video Segmentation (CVPR'23 Highlight)

[Ali Athar](https://www.aliathar.net/), Alexander Hermans, Jonathon Luiten, Deva Ramanan, Bastian Leibe

[`PDF`](https://arxiv.org/pdf/2301.02657.pdf) | [`Paperswithcode`](https://paperswithcode.com/paper/tarvis-a-unified-approach-for-target-based) | [`Cite`](https://github.com/Ali2500/TarViS/blob/main/README.md#cite)

**NOTE:** This repo and readme is a work-in-progress. I'll upload the code once the documentation is complete.

| Video Instance Segmentation | Video Panoptic Segmentation | Video Object Segmentation | Point Exemplar-guided Tracking |
| --- | --- | --- | --- |
| ![VIS](.images/vis.gif) | ![VOS](.images/vps.gif "VPS") | ![VOS](.images/vos.gif "VOS") | ![PET](.images/pet.gif "PET") |

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-instance-segmentation-on-youtube-vis-2)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-2?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-instance-segmentation-on-ovis-1)](https://paperswithcode.com/sota/video-instance-segmentation-on-ovis-1?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-panoptic-segmentation-on-kitti-step)](https://paperswithcode.com/sota/video-panoptic-segmentation-on-kitti-step?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-panoptic-segmentation-on-cityscapes-vps)](https://paperswithcode.com/sota/video-panoptic-segmentation-on-cityscapes-vps?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/visual-object-tracking-on-davis-2017)](https://paperswithcode.com/sota/visual-object-tracking-on-davis-2017?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/on-burst-point-exemplar-guided-val)](https://paperswithcode.com/sota/on-burst-point-exemplar-guided-val?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/on-burst-point-exemplar-guided-test)](https://paperswithcode.com/sota/on-burst-point-exemplar-guided-test?p=tarvis-a-unified-approach-for-target-based)

## Abstract

> The general domain of video segmentation is currently fragmented into different tasks spanning multiple benchmarks. Despite rapid progress in the state-of-the-art, current methods are overwhelmingly task-specific and cannot conceptually generalize to other tasks. Inspired by recent approaches with multi-task capability, we propose TarViS: a novel, unified network architecture that can be applied to any task that requires segmenting a set of arbitrarily def ined ‘targets’ in video. Our approach is flexible with respect to how tasks define these targets, since it models the latter as abstract ‘queries’ which are then used to predict pixel-precise target masks. A single TarViS model can be trained jointly on a collection of datasets spanning different tasks, and can hot-swap between tasks during inference without any task-specific retraining. To demonstrate its effectiveness, we apply TarViS to four different tasks, namely Video Instance Segmentation (VIS), Video Panoptic Segmentation (VPS), Video Object Segmentation (VOS) and Point Exemplar-guided Tracking (PET). Our unified, jointly trained model achieves state-of-the-art performance on 5/7 benchmarks spanning these four tasks, and competitive performance on the remaining two

## Directory Setup

For managing datasets, checkpoints and pretrained backbones, we use a single environment variable `$TARVIS_WORKSPACE_DIR` which points to a directory that is organized as follows:

```
├── $TARVIS_WORKSPACE_DIR
│   ├── checkpoints                   <- model weights saved here during training
│   ├── pretrained_backbones          <- ImageNet pretrained Swin backbone weights
│   ├── dataset_images                <- Images/videos for all datasets go here
|   |   ├── training
|   |   |   ├── ade20k               
|   |   |   |   ├── ADE_train_00000001.jpg
|   |   |   |   ├── ...
|   |   |   ├── cityscapes
|   |   |   |   ├── aachen
|   |   |   |   ├── ...
|   |   |   ├── coco
|   |   |   |   ├── 000000000009.jpg
|   |   |   |   ├── ...
|   |   |   ├── mapillary
|   |   |   |   ├── 0035fkbjWljhaftpVM37-g.jpg
|   |   |   |   ├── ...
|   |   |   ├── cityscapes_vps
|   |   |   |   ├── 0001_0001_frankfurt_000000_000279_newImg8bit.png
|   |   |   |   ├── ...
|   |   |   ├── kitti_step
|   |   |   |   ├── 0000
|   |   |   |   ├── ...
|   |   |   ├── vipseg
|   |   |   |   ├── 0_wHveSGjXyDY
|   |   |   |   ├── ...
|   |   |   ├── youtube_vis_2021
|   |   |   |   ├── 3245e049fb
|   |   |   |   ├── ...
|   |   |   ├── ovis
|   |   |   |   ├── 001ca3cb
|   |   |   |   ├── ...
|   |   |   ├── davis 
|   |   |   |   ├── bear
|   |   |   |   ├── ...
|   |   |   ├── burst   
|   |   |   |   ├── YFCC100M
|   |   |   |   ├── ...
|   |   ├── inference
|   |   |   ├── cityscapes_vps_val
|   |   |   ├── kitti_step_val
|   |   |   ├── vipseg
|   |   |   ├── youtube_vis_2021     
|   |   |   ├── ovis
|   |   |   ├── davis                 
|   |   |   ├── burst
|   |   |   |   ├── val
|   |   |   |   |   ├── YFCC100M
|   |   |   |   |   ├── ...
|   |   |   |   ├── test
|   |   |   |   |   ├── YFCC100M
|   |   |   |   |   ├── ...
|   ├── dataset_annotation            <- Annotations for all datasets go here
|   |   ├── training
|   |   |   ├── ade20k_panoptic    
|   |   |   |   ├── pan_maps
|   |   |   |   |   ├── ADE_train_00000001.png
|   |   |   |   ├── segments.json
|   |   |   ├── cityscapes_panoptic
|   |   |   |   ├── pan_maps
|   |   |   |   |   ├── aachen_000000_000019_gtFine_panoptic.png
|   |   |   |   |   ├── ...
|   |   |   |   ├── segments.json
|   |   |   ├── coco_panoptic
|   |   |   |   ├── pan_maps
|   |   |   |   |   ├── 000000000009.png
|   |   |   |   |   ├── ...
|   |   |   |   ├── segments.json
|   |   |   ├── mapillary_panoptic
|   |   |   |   ├── pan_maps    
|   |   |   |   |   ├── 0035fkbjWljhaftpVM37-g.png
|   |   |   |   |   ├── ...
|   |   |   |   ├── segments.json
|   |   |   ├── cityscapes_vps.json
|   |   |   ├── kitti_step.json
|   |   |   ├── vipseg
|   |   |   |   ├── panoptic_masks
|   |   |   |   |   ├── 0_wHveSGjXyDY
|   |   |   |   |   ├── ...
|   |   |   |   ├── video_info.json
|   |   |   ├── youtube_vis_2021.json
|   |   |   ├── ovis.json
|   |   |   ├── davis_semisupervised.json
|   |   |   ├── burst.json
|   |   ├── inference   
|   |   |   ├── cityscapes_vps
|   |   |   |   ├── im_all_info_val_city_vps.json
|   |   |   ├── vipseg
|   |   |   |   ├── val.json
|   |   |   ├── youtube_vis
|   |   |   |   ├── valid_2021.json
|   |   |   ├── ovis
|   |   |   |   ├── valid.json
|   |   |   ├── davis
|   |   |   |   ├── Annotations
|   |   |   |   ├── ImageSet_val.txt
|   |   |   |   ├── ImageSet_testdev.txt
|   |   |   ├── burst
|   |   |   |   ├── first_frame_annotations_val.json
|   |   |   |   ├── first_frame_annotations_test.json
```

Note that you do not need to setup all the datasets if you only want to train/infer on a sub-set of them. For training the full model however, you need the complete directory tree above

## Cite

```
@inproceedings{athar2023tarvis,
  title={TarViS: A Unified Architecture for Target-based Video Segmentation},
  author={Athar, Ali and Hermans, Alexander and Luiten, Jonathon and Ramanan, Deva and Leibe, Bastian},
  booktitle={CVPR},
  year={2023}
}
```
