"""
building-segmentation
Proof of concept showing effectiveness of a fine tuned instance segmentation model for deteting buildings.
"""
import os
import cv2
os.system("pip install 'git+https://github.com/facebookresearch/detectron2.git'")
from transformers import DetrFeatureExtractor, DetrForSegmentation
from PIL import Image
import gradio as gr
import numpy as np
import torch
import torchvision
import detectron2

# import some common detectron2 utilities
import itertools
import seaborn as sns
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer

cfg = get_cfg()
cfg.merge_from_file("model_weights/buildings_poc_cfg.yml")
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35
cfg.MODEL.WEIGHTS = "model_weights/model_final.pth"  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
predictor = DefaultPredictor(cfg)

def segment_buildings(im, confidence_threshold):
    im = np.array(im)
    outputs = predictor(im)

    instances = outputs["instances"].to("cpu")
    scores = instances.scores
    selected_indices = scores > confidence_threshold
    selected_instances = instances[selected_indices]

    v = Visualizer(im[:, :, ::-1],
                   scale=0.5,
                   instance_mode=ColorMode.SEGMENTATION
    )
    out = v.draw_instance_predictions(selected_instances)

    return Image.fromarray(out.get_image()[:, :, ::-1])

# gradio components 

gr_slider_confidence = gr.inputs.Slider(0,1,.1,.7,
                                        label='Set confidence threshold % for masks')

# gradio outputs
inputs = gr.inputs.Image(type="pil", label="Input Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

title = "Building Segmentation"
description = "An instance segmentation demo for identifying boundaries of buildings in aerial images using DETR (End-to-End Object Detection) model with MaskRCNN-101 backbone"

# Create user interface and launch
gr.Interface(segment_buildings, 
                inputs = [inputs, gr_slider_confidence],
                outputs = outputs,
                 title = title,
                description = description).launch(debug=True)