import cv2
import numpy as np


def bbox_from_mask(mask, order='Y1Y2X1X2', return_none_if_invalid=False):
    reduced_y = np.any(mask, axis=0)
    reduced_x = np.any(mask, axis=1)

    x_min = reduced_y.argmax()
    if x_min == 0 and reduced_y[0] == 0:  # mask is all zeros
        if return_none_if_invalid:
            return None
        else:
            return -1, -1, -1, -1

    x_max = len(reduced_y) - np.flip(reduced_y, 0).argmax()

    y_min = reduced_x.argmax()
    y_max = len(reduced_x) - np.flip(reduced_x, 0).argmax()

    if order == 'Y1Y2X1X2':
        return y_min, y_max, x_min, x_max
    elif order == 'X1X2Y1Y2':
        return x_min, x_max, y_min, y_max
    elif order == 'X1Y1X2Y2':
        return x_min, y_min, x_max, y_max
    elif order == 'Y1X1Y2X2':
        return y_min, x_min, y_max, x_max
    else:
        raise ValueError("Invalid order argument: %s" % order)


def parse_bbox(bbox, mask):
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()

    if isinstance(bbox, (list, tuple)):
        assert len(bbox) == 4, f"Bounding box must have length 4, but got array of shape {bbox.shape}"
        return bbox

    elif bbox == "mask":
        assert mask is not None
        bbox = bbox_from_mask(mask, order='X1Y1X2Y2', return_none_if_invalid=True)
        return bbox

    elif bbox is not None:
        raise ValueError(f"Invalid bounding box provided: {bbox}")

    return None


def handle_cv2umat(img):
    if isinstance(img, cv2.UMat):
        return img.get()

    else:
        return img


# from DAVIS dataset API
def create_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_mask_on_image(image, mask, mask_opacity=0.6, mask_color=(0, 255, 0)):
    if mask.ndim == 3:
        assert mask.shape[2] == 1
        _mask = mask.squeeze(axis=2)
    else:
        _mask = mask
    mask_bgr = np.stack((_mask, _mask, _mask), axis=2)
    masked_image = np.where(mask_bgr > 0, mask_color, image)
    return ((mask_opacity * masked_image) + ((1. - mask_opacity) * image)).astype(np.uint8)


def annotate_image_instance(image, mask=None, color=None, label=None, bbox=None, **kwargs):
    """
    :param image: np.ndarray(H, W, 3)
    :param mask: np.ndarray(H, W)
    :param color: tuple/list(int, int, int) in range [0, 255]
    :param label: str
    :param bbox: either "none" or "mask" or tuple of 4 box coords in xyxy format
    :param kwargs: "bbox_thickness", "text_font", "font_size", "mask_opacity", "text_placement"
    :return: np.ndarray(H, W, 3)
    """
    # parse box
    bbox_generated_from_mask = bbox == "mask"
    bbox = parse_bbox(bbox, mask)
    assert not (bbox is None and mask is None)

    # parse color
    if color is None:
        color = (0, 255, 0)  # green
    else:
        color = tuple(color)  # cv2 does not like colors as lists

    # draw mask
    if mask is None:
        annotated_image = np.copy(image)
    else:
        assert image.shape[:2] == mask.shape, f"Shape mismatch between image {image.shape} and mask {mask.shape}"
        annotated_image = overlay_mask_on_image(image, mask, mask_color=color, mask_opacity=kwargs.get("mask_opacity", 0.6))

    # if no box and label are present then we're done
    if bbox is None and label is None:
        return annotated_image

    # if box is valid then draw it
    if bbox is not None:
        bbox_thickness = kwargs.get("bbox_thickness", 2)
        xmin, ymin, xmax, ymax = map(int, bbox)
        annotated_image = cv2.rectangle(cv2.UMat(annotated_image), (xmin, ymin), (xmax, ymax), color=tuple(color),
                                        thickness=bbox_thickness)

    # draw mask labels
    if label is not None:
        text_font = kwargs.get("text_font", cv2.FONT_HERSHEY_SIMPLEX)
        font_size = kwargs.get("font_size", 0.5)
        font_thickness = kwargs.get("font_thickness", 1)
        font_line_type = kwargs.get("font_line_type", cv2.LINE_AA)

        (text_width, text_height), _ = cv2.getTextSize(label, text_font, font_size, thickness=1)

        text_offset_x, text_offset_y = None, None

        text_placement = kwargs.get("text_placement", "topleft")
        if text_placement == "topleft":
            # for this a bounding-box is required. We have 2 options:
            # (1) If an all-zeros mask was provided and bbox="mask", then we do nothing
            # (2) If bbox=None then we throw an error
            if bbox is None:
                if bbox_generated_from_mask and not np.any(mask):  # (1)
                    pass
                else:  # (2)
                    assert bbox is not None, f"For text placement mode 'topleft', a bounding box must be present"
            else:
                xmin, ymin, xmax, ymax = map(int, bbox)
                text_offset_x, text_offset_y = int(xmin + 2), int(ymin + text_height + 2)

        elif text_placement == "mask_centroid":
            assert mask is not None, f"Mask must be provided for text placement mode 'mask_centroid'"
            mask_pts_y, mask_pts_x = np.nonzero(mask)

            if len(mask_pts_y) > 0:
                text_offset_x = (mask_pts_x.astype(np.float32).mean() - (text_width / 2.0)).round().astype(int).item()
                text_offset_y = (mask_pts_y.astype(np.float32).mean() + (text_height / 2.0)).round().astype(int).item()

        else:
            raise ValueError(f"Invalid 'text_placement' argument: {text_placement}")

        if text_offset_x is None:
            return handle_cv2umat(annotated_image)

        if kwargs.get("text_in_white_box", True):
            text_bg_box_pt1 = int(text_offset_x), int(text_offset_y + 2)
            text_bg_box_pt2 = int(text_offset_x + text_width + 2), int(text_offset_y - text_height - 2)
            annotated_image = cv2.rectangle(cv2.UMat(annotated_image), text_bg_box_pt1, text_bg_box_pt2, color=(255, 255, 255), thickness=-1)

        text_color = tuple(kwargs.get("text_color", (0, 0, 0)))
        annotated_image = cv2.putText(cv2.UMat(annotated_image), label, (text_offset_x, text_offset_y),
                                      fontFace=text_font,
                                      fontScale=font_size,
                                      color=text_color,
                                      thickness=font_thickness,
                                      lineType=font_line_type)

    # sometimes OpenCV will retun an object of type cv2.UMat instead of a numpy array (don't know why)
    return handle_cv2umat(annotated_image)
