import torch
import cv2
import numpy as np
from numpy.linalg import norm

from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from data.config import cfg_re50

import onnxruntime as ort

class RetinaFace:
    def __init__(self, model_path, cfg, input_size = (640, 640), device = 'cpu'):
        self.input_size = input_size
        self.cfg = cfg
        self.device = device

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.priorbox = PriorBox(cfg, image_size=input_size)
        self.priors = self.priorbox.forward()
        self.priors = torch.from_numpy(self.priors).to(device)
        self.scale = torch.Tensor([input_size[1], input_size[0], input_size[1], input_size[0]], device = self.device)