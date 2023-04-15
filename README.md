# TarViS: A Unified Architecture for Target-based Video Segmentation (CVPR'23 Highlight)

[Ali Athar](https://www.aliathar.net/), Alexander Hermans, Jonathon Luiten, Deva Ramanan, Bastian Leibe

[`PDF`](https://arxiv.org/pdf/2301.02657.pdf) | [`Cite`](https://github.com/Ali2500/TarViS/blob/main/README.md#cite)

| Video Instance Segmentation | Video Panoptic Segmentation | Video Object Segmentation | Point Exemplar-guided Tracking |
| --- | --- | --- | --- |
| ![VIS](.images/vis.gif) | ![VOS](.images/vps.gif "VPS") | ![VOS](.images/vos.gif "VOS") | ![PET](.images/pet.gif "PET") |

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-instance-segmentation-on-youtube-vis-2)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-2?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-instance-segmentation-on-ovis-1)](https://paperswithcode.com/sota/video-instance-segmentation-on-ovis-1?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-panoptic-segmentation-on-kitti-step)](https://paperswithcode.com/sota/video-panoptic-segmentation-on-kitti-step?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/video-panoptic-segmentation-on-cityscapes-vps)](https://paperswithcode.com/sota/video-panoptic-segmentation-on-cityscapes-vps?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/visual-object-tracking-on-davis-2017)](https://paperswithcode.com/sota/visual-object-tracking-on-davis-2017?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/on-exemplar-guided-point-val)](https://paperswithcode.com/sota/on-exemplar-guided-point-val?p=tarvis-a-unified-approach-for-target-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarvis-a-unified-approach-for-target-based/on-exemplar-guided-point-test)](https://paperswithcode.com/sota/on-exemplar-guided-point-test?p=tarvis-a-unified-approach-for-target-based)

## Abstract

> The general domain of video segmentation is currently fragmented into different tasks spanning multiple benchmarks. Despite rapid progress in the state-of-the-art, current methods are overwhelmingly task-specific and cannot conceptually generalize to other tasks. Inspired by recent approaches with multi-task capability, we propose TarViS: a novel, unified network architecture that can be applied to any task that requires segmenting a set of arbitrarily def ined ‘targets’ in video. Our approach is flexible with respect to how tasks define these targets, since it models the latter as abstract ‘queries’ which are then used to predict pixel-precise target masks. A single TarViS model can be trained jointly on a collection of datasets spanning different tasks, and can hot-swap between tasks during inference without any task-specific retraining. To demonstrate its effectiveness, we apply TarViS to four different tasks, namely Video Instance Segmentation (VIS), Video Panoptic Segmentation (VPS), Video Object Segmentation (VOS) and Point Exemplar-guided Tracking (PET). Our unified, jointly trained model achieves state-of-the-art performance on 5/7 benchmarks spanning these four tasks, and competitive performance on the remaining two

Code release coming soon


## Cite

```
@inproceedings{athar2023tarvis,
  title={TarViS: A Unified Architecture for Target-based Video Segmentation},
  author={Athar, Ali and Hermans, Alexander and Luiten, Jonathon and Ramanan, Deva and Leibe, Bastian},
  booktitle={CVPR},
  year={2023}
}
```
