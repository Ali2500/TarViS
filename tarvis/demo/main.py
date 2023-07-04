from argparse import ArgumentParser
import dash
import plotly.express as px
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import numpy as np
from PIL import Image
import tempfile
from io import BytesIO

import cv2
import os.path as osp
import torch

from tarvis.demo.utils import (
    shape_to_png, 
    rgb_to_hex, 
    mask_overlap_check, 
    base64_to_image,
    write_image_sequence_as_video,
    video_content_to_frame_list,
    video_path_to_bytes
)
from tarvis.demo.inferer import TarvisDemoInferer
from tarvis.utils.visualization import overlay_mask_on_image, create_color_map
from tarvis.inference.tarvis_inference_model import TarvisInferenceModel
from tarvis.utils.paths import Paths
from tarvis.config import cfg


INFERER = TarvisDemoInferer()
COLOR_MAP = create_color_map().tolist()[1:]
SEQ_IMAGES = []
IMAGE_HEIGHT = None
IMAGE_WIDTH = None
FIRST_FRAME_IMAGE = None
CURRENT_ANN_INDEX = 0
MASK_ANNS = []
POINT_ANNS = []

default_video_path = "assets/video.mp4"
default_image_frame_paths = [f"tarvis/demo/assets/{str(x).zfill(5)}.jpg" for x in range(0, 50)]

popup_dialog = dcc.ConfirmDialog(id='popup')

heading = html.Div(html.H1("TarViS Demo", style={'color': '#10184E', 'background-color': '#D9D9D9', 'text-align': 'center'}))
step1 = html.Div([html.H4("Step 1: Upload a video", style={'margin-left': '10px'}), 
                  html.Div("You have three options: (1) Select an MP4 video files. (2) Select a sequence of PNG/JPG image files. (3) Do nothing and just use the pre-loaded video shown below.", style={'margin-left': '10px'})])


video_upload = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Video/Image Files')
        ]),
        style={
            'width': '70%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'margin-top': '10px',
            'margin-bottom': '10px',
            'background-color': '#E3E3E3'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )

message_box = html.Div(id='message-box')


video_player_text = html.Div(html.H5("Video Preview:", style={'margin-left': '10px'}))
video_player = html.Div(
    style={"width": "70%", "padding": "0px", 'margin-left': 'auto', 'margin-right': 'auto'},
    children=[
        html.Video(
            id="video-player",
            src=default_video_path,
            controls=True,
            width="100%",
            height="320px",
    )],
)

step2 = html.Div([html.Hr(), html.H4("Step 2: Select a task", style={'margin-left': '10px'}), 
                  html.Div(["You have two options:", html.Br(), "1) Instance/panoptic segmentation using classes from one of the datasets", html.Br(), "2) VOS/PET by manually annotating the first-frame masks/points for the objects that you want to segment."], style={'margin-left': '10px'})])

dropdown_obj = html.Div(dcc.Dropdown([
    "VOS          (annotate masks)",
    "PET          (annotate points)",
    "Instance Seg (YouTube-VIS)",
    "Instance Seg (OVIS)",
    "Panoptic Seg (KITTI-STEP)",
    "Panoptic Seg (Cityscapes-VPS)",
    "Panoptic Seg (VIPSeg)"
], id='task-dropdown', clearable=False, placeholder="Select a task..."))
run_button = html.Div(html.Button("Run!", id='run-button'), style={'margin-left': '10px'})

task_dropdown = dbc.Row([dbc.Col(dropdown_obj, width=4), dbc.Col(run_button, width='auto')], style={'margin': '10px'})

SEQ_IMAGES = [np.array(Image.open(path)) for path in default_image_frame_paths]
FIRST_FRAME_IMAGE = np.copy(SEQ_IMAGES[0])
IMAGE_HEIGHT, IMAGE_WIDTH = FIRST_FRAME_IMAGE.shape[:2]
ff_fig = px.imshow(FIRST_FRAME_IMAGE, binary_string=True)
ff_fig.update_layout({"dragmode": "drawclosedpath", "newshape.line.color": rgb_to_hex(*COLOR_MAP[CURRENT_ANN_INDEX]), "height": 640})

mask_draw_div = html.Div(
    children=[dcc.Graph(id="graph-pic-camera", figure=ff_fig)],
    style={"width": "100%", "display": "inline-block", "padding": "0 0", "margin-left": "auto", "margin-right": "auto"}, hidden=True, id='mask-draw-div'
)

ff_fig_pt = px.imshow(FIRST_FRAME_IMAGE, binary_string=True)
ff_fig_pt.update_layout({"height": 640, "dragmode": "select"})

point_draw_div = html.Div(
    children=[dcc.Graph(id="point-draw-graph", figure=ff_fig_pt)],
    style={"width": "100%", "display": "inline-block", "padding": "0 0", "margin-left": "auto", "margin-right": "auto"}, hidden=True, id='point-draw-div'
)

# run_button = html.Div(
#     [html.Button("Run!", id='run-button')], 
#     style={"margin-left": "auto", "margin-right": "auto", "margin-top": "10px", "text-align": "center"}
# )

result_video_player_text = html.Div([html.Hr(), html.H5("Output:", style={'margin-left': '10px'})])
result_video_player = html.Div(
    style={"width": "70%", "padding": "0px", 'margin-left': 'auto', 'margin-right': 'auto'},
    children=[
        html.Video(
            id="result-video-player",
            src=None,
            controls=True,
            width="100%",
            height="320px",
    )],
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder="/home/athar/my_github_repos/TarViS/tarvis/demo/assets")
app.title = "TarViS Demo"
app.layout = html.Div([
    popup_dialog, heading, step1, video_upload, message_box, video_player_text, video_player, step2, 
    task_dropdown, mask_draw_div, point_draw_div, result_video_player_text, result_video_player], id='main'
)
    

@callback(
        [Output('mask-draw-div', 'hidden'), 
         Output('graph-pic-camera', 'figure', allow_duplicate=True), 
         Output('point-draw-div', 'hidden'), 
         Output('point-draw-graph', 'figure', allow_duplicate=True)],
        [Input('task-dropdown', 'value')],
        prevent_initial_call=True
)
def task_choice(task_value):
    global POINT_ANNS, FIRST_FRAME_IMAGE, CURRENT_ANN_INDEX

    if task_value is None:
        return [True, dash.no_update, True, dash.no_update]
    
    if "PET" in task_value:
        CURRENT_ANN_INDEX = 0
        MASK_ANNS = []
        FIRST_FRAME_IMAGE = np.copy(SEQ_IMAGES[0])
        fig = px.imshow(FIRST_FRAME_IMAGE, binary_string=True)
        fig.update_layout({"height": 640, "dragmode": "select"})
        return [True, dash.no_update, False, fig]
    
    elif "VOS" in task_value:
        CURRENT_ANN_INDEX = 0
        POINT_ANNS = []
        FIRST_FRAME_IMAGE = np.copy(SEQ_IMAGES[0])
        fig = px.imshow(FIRST_FRAME_IMAGE, binary_string=True)
        fig.update_layout({"dragmode": "drawclosedpath", "newshape.line.color": rgb_to_hex(*COLOR_MAP[CURRENT_ANN_INDEX]), "height": 640})
        return [False, fig, True, dash.no_update]
    
    else:
        return [True, dash.no_update, True, dash.no_update]

@callback([Output('graph-pic-camera', 'figure'),
           Output('video-player', 'src'), 
           Output('popup', 'message', allow_duplicate=True),
           Output('popup', 'displayed', allow_duplicate=True)],
          [Input('upload-data', 'contents'),
           State('upload-data', 'filename')], prevent_initial_call=True)
def process_selected_video(list_of_contents, list_of_names):
    print(list_of_names)

    if list_of_names is None:
        raise dash.exceptions.PreventUpdate()
    
    images = []
    
    if list_of_names[0].endswith(".mp4"):
        if len(list_of_names) > 1:
            updated_msg = "Only a single video file may be selected"
            return [dash.no_update, dash.no_update, updated_msg, True]   
        else:
            images = video_content_to_frame_list(list_of_contents[0])
            video_bytes = dash.no_update

    elif list_of_names[0].endswith(".png") or list_of_names[0].endswith(".jpg"):
        # sort according to filename
        sort_idx = np.argsort(list_of_names).tolist()
        list_of_names = [list_of_names[i] for i in sort_idx]
        list_of_contents = [list_of_contents[i] for i in sort_idx]
        images = [base64_to_image(image_str) for image_str in list_of_contents]

        if len(images) == 1:
            updated_msg = "Multiple image files must be selected"
            return [dash.no_update, dash.no_update, updated_msg, True]
        
        if not all([im.shape[:2] == images[0].shape[:2] for im in images]):
            updated_msg = "All images must have the same shape"
            return [dash.no_update, dash.no_update, updated_msg, True]
        
        temp_vid_path = osp.join(tempfile.mkdtemp(), "video.mp4")
        write_image_sequence_as_video(images, temp_vid_path)
        video_bytes = video_path_to_bytes(temp_vid_path)
        updated_msg = f"Selected {len(images)} image files"

    img = images[0]

    global IMAGE_HEIGHT, IMAGE_WIDTH, FIRST_FRAME_IMAGE, CURRENT_ANN_INDEX, MASK_ANNS, POINT_ANNS, SEQ_IMAGES
    IMAGE_HEIGHT, IMAGE_WIDTH = img.shape[:2]
    FIRST_FRAME_IMAGE = np.copy(img)
    CURRENT_ANN_INDEX = 0
    MASK_ANNS = []
    POINT_ANNS = []
    SEQ_IMAGES = images

    fig = px.imshow(img, binary_string=True)
    fig.update_layout({"dragmode": "drawclosedpath", "newshape.line.color": rgb_to_hex(*COLOR_MAP[CURRENT_ANN_INDEX])})
    
    return [fig, video_bytes, dash.no_update, dash.no_update]


@callback(
        [Output('graph-pic-camera', 'figure', allow_duplicate=True), 
         Output('popup', 'message', allow_duplicate=True),
         Output('popup', 'displayed', allow_duplicate=True)], 
        [Input('graph-pic-camera', 'relayoutData'), 
         State('graph-pic-camera', 'figure')], prevent_initial_call=True)
def mask_drawn(graph_relayout, current_fig):
    if "dragmode" in graph_relayout:
        raise dash.exceptions.PreventUpdate()

    if "shapes" not in graph_relayout:
        current_fig["layout"]["dragmode"] = "drawclosedpath"
        return [current_fig, dash.no_update, dash.no_update]
    
    global CURRENT_ANN_INDEX, MASK_ANNS, FIRST_FRAME_IMAGE
    
    shape_args = [
        {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "shape": shape}
        for shape in graph_relayout["shapes"]
    ]
    shape_args = [shape_args[-1]]

    images = []
    for sa in shape_args:
        pngbytes = shape_to_png(**sa)
        mask = np.array(Image.open(BytesIO(pngbytes)))
        mask = np.any(mask, 2)
        images.append(mask)

    combined_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.bool)
    for m in images:
        combined_mask = np.logical_or(combined_mask, m)

    combined_mask = combined_mask.astype(np.uint8)
    MASK_ANNS.append(np.copy(combined_mask))      
    if mask_overlap_check(MASK_ANNS):
        FIRST_FRAME_IMAGE = overlay_mask_on_image(FIRST_FRAME_IMAGE, combined_mask, 0.5, mask_color=COLOR_MAP[CURRENT_ANN_INDEX], border_thickness=5)
        CURRENT_ANN_INDEX += 1
        msg, show = dash.no_update, dash.no_update
    else:
        MASK_ANNS.pop()
        msg = "The mask overlaps with one or more existing masks. Please redraw it without any overlap"
        show = True

    fig = px.imshow(FIRST_FRAME_IMAGE, binary_string=True)
    fig.update_layout({"dragmode": "drawclosedpath", "newshape.line.color": rgb_to_hex(*COLOR_MAP[CURRENT_ANN_INDEX])})
    return [fig, msg, show]


@callback(
        [Output('point-draw-graph', 'figure', allow_duplicate=True), Output('message-box', 'children', allow_duplicate=True)], 
        [Input('point-draw-graph', 'clickData'), State('message-box', 'children'), State('point-draw-graph', 'figure')], prevent_initial_call=True)
def point_drawn(graph_click_data, current_msg, current_fig):
    print(graph_click_data)
    pt_x = graph_click_data["points"][0]['x']
    pt_y = graph_click_data["points"][0]['y']
    global POINT_ANNS, FIRST_FRAME_IMAGE, CURRENT_ANN_INDEX

    for curr_y, curr_x in POINT_ANNS:
        if curr_y == pt_y and curr_x == pt_x:
            updated_msg = f"The selected point ({curr_y}, {curr_x}) has already been used before"
            return [dash.no_update, updated_msg]

    POINT_ANNS.append((pt_y, pt_x))
    FIRST_FRAME_IMAGE = cv2.circle(FIRST_FRAME_IMAGE, (pt_x, pt_y), 6, color=COLOR_MAP[CURRENT_ANN_INDEX], thickness=-1)
    FIRST_FRAME_IMAGE = cv2.circle(FIRST_FRAME_IMAGE, (pt_x, pt_y), 7, color=(0, 0, 0), thickness=2)
    CURRENT_ANN_INDEX += 1

    fig = px.imshow(FIRST_FRAME_IMAGE, binary_string=True)
    fig.update_layout({"height": 640, "dragmode": "select"})

    return [fig, dash.no_update]


@callback(
        [Output('result-video-player', 'src'),
         Output('popup', 'message', allow_duplicate=True),
         Output('popup', 'displayed', allow_duplicate=True)],
        [Input('run-button', 'n_clicks'),
         State('task-dropdown', 'value')], prevent_initial_call=True
)
def run_button_pressed(n_clicks, task_value):
    global INFERER, SEQ_IMAGES, MASK_ANNS, POINT_ANNS

    if "VOS" in task_value and not MASK_ANNS:
        return [dash.no_update, "One or more mask annotations are required", True]
    elif "PET" in task_value and not POINT_ANNS:
        return [dash.no_update, "One or more point annotations are required", True]

    viz_images = INFERER.run(
        images=SEQ_IMAGES,
        task_type=task_value,
        first_frame_masks=MASK_ANNS,
        first_frame_points=POINT_ANNS
    )
    
    temp_vid_path = osp.join(tempfile.mkdtemp(), "video.mp4")
    write_image_sequence_as_video(viz_images, temp_vid_path)
    video_bytes = video_path_to_bytes(temp_vid_path)

    return [video_bytes, dash.no_update, dash.no_update]


def main(args):
    global INFERER

    if not osp.isabs(args.model_path):
        args.model_path = osp.join(Paths.saved_models_dir(), args.model_path)

    expected_cfg_path = osp.join(osp.dirname(args.model_path), "config.yaml")
    assert osp.exists(expected_cfg_path), f"Config file not found at expected path: {expected_cfg_path}"
    cfg.merge_from_file(expected_cfg_path)

    print("Creating model...")
    INFERER.model = TarvisInferenceModel("YOUTUBE_VIS").cuda().eval()  # dataset name is a dummy value
    INFERER.model.restore_weights(args.model_path)

    app.run(debug=False)


if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("model_path")
    main(parser.parse_args())
