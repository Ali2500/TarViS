import os
import os.path as osp


def _get_env_var(name):
    if name not in os.environ:
        raise EnvironmentError("Required environment variable '{}' is not set".format(name))
    return os.environ[name]


class Paths(object):
    def __init__(self):
        raise ValueError("Static class 'Paths' should be not be initialized")

    # ------------------------------------------- TOP LEVEL DIRECTORIES -----------------------------------------------
    @classmethod
    def configs_dir(cls):
        return osp.realpath(osp.join(osp.dirname(__file__), os.pardir, os.pardir, "configs"))

    @classmethod
    def workspace_dir(cls):
        return osp.join(_get_env_var("TARVIS_WORKSPACE_DIR"))

    @classmethod
    def annotations_dir(cls):
        return osp.join(cls.workspace_dir(), "dataset_annotations")

    @classmethod
    def dataset_images_dir(cls):
        return osp.join(cls.workspace_dir(), "dataset_images")

    @classmethod
    def training_dataset_images_dir(cls):
        return osp.join(cls.dataset_images_dir(), "training")

    @classmethod
    def inference_dataset_images_dir(cls):
        return osp.join(cls.dataset_images_dir(), "inference")

    @classmethod
    def training_annotations_dir(cls):
        return osp.join(cls.annotations_dir(), "training")

    @classmethod
    def inference_annotations_dir(cls):
        return osp.join(cls.annotations_dir(), "inference")

    @classmethod
    def saved_models_dir(cls):
        return osp.join(cls.workspace_dir(), "checkpoints")

    @classmethod
    def pretrained_backbones_dir(cls):
        return osp.join(cls.workspace_dir(), "pretrained_backbones")

    # --------------------------------------- IMAGE-LEVEL PANOPTIC DATASETS --------------------------------
    @classmethod
    def panoptic_train_images(cls, dataset_name):
        return osp.join(cls.training_dataset_images_dir(), dataset_name)

    @classmethod
    def panoptic_train_anns(cls, dataset_name):
        pan_maps = osp.join(cls.training_annotations_dir(), f"{dataset_name}_panoptic", "pan_maps")
        segments_json = osp.join(cls.training_annotations_dir(), f"{dataset_name}_panoptic", "segments.json")
        return pan_maps, segments_json    

    # ------------------------------------------- YOUTUBE-VIS -----------------------------------------------
    @classmethod
    def youtube_vis_train_images(cls):
        return osp.join(cls.training_dataset_images_dir(), "youtube_vis_2021")

    @classmethod
    def youtube_vis_train_anns(cls):
        return osp.join(cls.training_annotations_dir(), "youtube_vis_2021.json")

    @classmethod
    def youtube_vis_val_paths(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "youtube_vis_2021"),
            "json_info_path": osp.join(cls.inference_annotations_dir(), "youtube_vis", "valid_2021.json")
        }

    # ----------------------------------------------- OVIS --------------------------------------------------
    @classmethod
    def ovis_train_images(cls):
        # return osp.join(cls.training_dataset_images_dir(), "ovis")
        return osp.join(cls.training_dataset_images_dir(), "ovis")

    @classmethod
    def ovis_train_anns(cls):
        return osp.join(cls.training_annotations_dir(), "ovis.json")

    @classmethod
    def ovis_val_paths(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "ovis"),
            "json_info_path": osp.join(cls.inference_annotations_dir(), "ovis", "valid.json")
        }

    # --------------------------------------------- DAVIS ---------------------------------------------------
    @classmethod
    def davis_train_images(cls):
        return osp.join(cls.training_dataset_images_dir(), "davis")

    @classmethod
    def davis_train_anns(cls):
        return osp.join(cls.training_annotations_dir(), "davis_semisupervised.json")

    @classmethod
    def davis_val_paths(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "davis"),
            "annotations_base_dir": osp.join(cls.inference_annotations_dir(), "davis", "Annotations"),
            "image_set_file_path": osp.join(cls.inference_annotations_dir(), "davis", "ImageSet_val.txt")
        }

    @classmethod
    def davis_testdev_paths(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "davis"),
            "annotations_base_dir": osp.join(cls.inference_annotations_dir(), "davis", "Annotations"),
            "image_set_file_path": osp.join(cls.inference_annotations_dir(), "davis", "ImageSet_testdev.txt")
        }

    # --------------------------------------------- BURST ---------------------------------------------------
    @classmethod
    def burst_train_images(cls):
        return osp.join(cls.training_dataset_images_dir(), "burst")

    @classmethod
    def burst_train_anns(cls):
        return osp.join(cls.training_annotations_dir(), "burst.json")

    @classmethod
    def burst_val_anns(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "burst", "val"),
            "first_frame_annotations_file": osp.join(cls.inference_annotations_dir(), "burst", "first_frame_annotations_val.json")
        }

    @classmethod
    def burst_test_anns(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "burst", "test"),
            "first_frame_annotations_file": osp.join(cls.inference_annotations_dir(), "burst", "first_frame_annotations_test.json")
        }

    # -------------------------------------------- KITTI-STEP -----------------------------------------------
    @classmethod
    def kitti_step_train_images(cls):
        return osp.join(cls.training_dataset_images_dir(), "kitti_step")

    @classmethod
    def kitti_step_train_anns(cls):
        return osp.join(cls.training_annotations_dir(), "kitti_step.json")

    @classmethod
    def kitti_step_val_paths(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "kitti_step_val")
        }

    # ------------------------------------------ CITYSCAPES-VPS ----------------------------------------------
    @classmethod
    def cityscapes_vps_train_images(cls):
        return osp.join(cls.training_dataset_images_dir(), "cityscapes_vps")

    @classmethod
    def cityscapes_vps_train_anns(cls):
        return osp.join(cls.training_annotations_dir(), "cityscapes_vps.json")

    @classmethod
    def cityscapes_vps_val_paths(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "cityscapes_vps_val"),
            "info_file": osp.join(cls.inference_annotations_dir(), "cityscapes_vps", "im_all_info_val_city_vps.json")
        }

    # ------------------------------------------ VIPSEG -------------------------------------------------------
    @classmethod
    def vipseg_train_images(cls):
        return osp.join(cls.training_dataset_images_dir(), "vipseg")

    @classmethod
    def vipseg_train_panoptic_masks(cls):
        return osp.join(cls.training_annotations_dir(), "vipseg", "panoptic_masks")

    @classmethod
    def vipseg_train_video_info(cls):
        return osp.join(cls.training_annotations_dir(), "vipseg", "video_info.json")

    @classmethod
    def vipseg_val_paths(cls):
        return {
            "images_base_dir": osp.join(cls.inference_dataset_images_dir(), "vipseg"),
            "panoptic_gt_json_file": osp.join(cls.inference_annotations_dir(), "vipseg", "val.json")
        }
