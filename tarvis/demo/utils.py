from cairosvg import svg2png
from typing import List
import moviepy.editor as mve
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from decord import VideoReader


def image_to_base64(image: np.ndarray):
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/png;base64, " + img_str.decode("utf-8")


def base64_to_image(base64_str: str):
    encoded_image = base64_str.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    return np.array(Image.open(bytes_image).convert('RGB'))


def write_image_sequence_as_video(images: List[np.ndarray], outpath: str):
    clips = [mve.ImageClip(im).set_duration(1.0/6.0) for im in images]
    concat_clip = mve.concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outpath, fps=6)


def video_path_to_bytes(vidpath: str):
    with open(vidpath, 'rb') as fh:
        content = fh.read()
    return "data:video/mp4;base64, " + base64.b64encode(content).decode()


def video_content_to_frame_list(content: str):
    encoded_video = content.split(",")[1]
    decoded_video = base64.b64decode(encoded_video)
    bytes_video = BytesIO(decoded_video)
    container = VideoReader(bytes_video)
    return [frame.asnumpy() for frame in container]


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def shape_to_svg_code(shape, width, height):
    fmt_dict = dict(
        width=width,
        height=height,
        stroke_color=shape["line"]["color"],
        stroke_width=shape["line"]["width"],
        path=shape["path"],
    )
    return """
<svg
    width="{width}"
    height="{height}"
    viewBox="0 0 {width} {height}"
>
<path
    stroke="{stroke_color}"
    stroke-width="{stroke_width}"
    d="{path}"
    fill-opacity="1"
/>
</svg>
""".format(
        **fmt_dict
    )


def shape_to_png(shape, width, height):
    """
    Like svg2png, if write_to is None, returns a bytestring. If it is a path
    to a file it writes to this file and returns None.
    """
    svg_code = shape_to_svg_code(shape, width=width, height=height)
    r = svg2png(bytestring=svg_code, write_to=None)
    return r


def mask_overlap_check(masks: List[np.ndarray]):
    mask_sum = np.zeros_like(masks[0], np.int64)
    for m in masks:
        mask_sum += m.astype(np.int64)
        if np.any(mask_sum > 1):
            return False
    return True
