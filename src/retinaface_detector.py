import torch
import numpy as np
import cv2

from layers.functions.prior_box import PriorBox
from data.config import cfg_re50
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.image_utils import resizeWithAspectRatio
from src.base_model import ModelLoader

class RetinaFaceDetector(ModelLoader):
    """ RetinaFace Detector using ONNX model """

    def __init__(self, model_path, device, input_size = (640, 640), cfg = cfg_re50, conf_thresh = 0.6, nms_thresh = 0.5):
        super().__init__(model_path, device)
        self.input_size = input_size
        self.cfg = cfg
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.priors = PriorBox(cfg, image_size=self.input_size).forward().to(self.device)
    
    def preprocessing(self, img) -> torch.Tensor:
        img = np.float32(img)
        img -= (104, 117, 123)                  # Mean subtraction
        img = img.transpose(2, 0, 1)            # HWC -> CHW
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img
    
    def detect(self, img: np.array):
        """ Detect faces in the input image.
        Args:
            img (numpy.ndarray): Input image.
        Returns:
            detections (numpy.ndarray): Detected bounding boxes.
            landms (numpy.ndarray): Detected landmarks.
            scores (numpy.ndarray): Confidence scores.
        """

        # Preprocess image and prepare input
        resized_img, scale, pad_left, pad_top, new_width, new_height = resizeWithAspectRatio(img, self.input_size)

        print(f"scale={scale:.3f}, pad_left={pad_left}, pad_top={pad_top}")

        img_height, img_width, _ = resized_img.shape

        img_tensor = self.preprocessing(resized_img)

        img_numpy = img_tensor.cpu().numpy()

        # Run ONNX model
        loc, conf, landms = self.infer(img_numpy)

        # Convert to tensor
        loc = torch.from_numpy(loc).to(self.device)
        conf = torch.from_numpy(conf).to(self.device)
        landms = torch.from_numpy(landms).to(self.device)

        # Decode bounding box and landmarks
        boxes, scores = self.decodeBoxes(loc, conf, torch.tensor([img_width, img_height, img_width, img_height], device=self.device))
        landms = self.decodeLandmarks(landms, new_width, new_height)
        
        # Apply Confidence Threshold
        boxes, landms, scores = self.applyConfidenceThreshold(boxes, landms, scores)
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)

        # # Scale back to original image size
        detections[:, [0, 2]] = (detections[:, [0, 2]] - pad_left) / scale
        detections[:, [1, 3]] = (detections[:, [1, 3]] - pad_top) / scale

        landms[:, 0::2] = (landms[:, 0::2] - pad_left) / scale
        landms[:, 1::2] = (landms[:, 1::2] - pad_top) / scale


        # Apply NMS
        keep = py_cpu_nms(detections, self.nms_thresh)
        detections = detections[keep, :]
        landms = landms[keep, :]

        return detections, landms, scores
    
    # def detect(self, img: np.array): 
    #     """ Detect faces in the input image. 
    #     Args:
    #         img (numpy.ndarray): Input image. 
    #     Returns: 
    #         detections (numpy.ndarray): Detected bounding boxes. 
    #         landms (numpy.ndarray): Detected landmarks. 
    #         scores (numpy.ndarray): Confidence scores. """ 
        
    #     # Preprocess image and prepare input 
    #     img = cv2.resize(img, self.input_size) 
    #     img_height, img_width, _ = img.shape 
    #     scale = torch.tensor([img_width, img_height, img_width, img_height], device = self.device) 
        
    #     img_tensor = self.preprocessing(img) 
    #     img_numpy = img_tensor.cpu().numpy() 
        
    #     # Run ONNX model
    #     loc, conf, landms = self.infer(img_numpy) 
        
    #     # Convert to tensor
    #     loc = torch.from_numpy(loc).to(self.device) 
    #     conf = torch.from_numpy(conf).to(self.device) 
    #     landms = torch.from_numpy(landms).to(self.device) 
        
    #     # Decode bounding box and landmarks
    #     boxes, scores = self.decodeBoxes(loc, conf, scale) 
    #     landms = self.decodeLandmarks(landms, img_width, img_height) 
        
    #     # Apply Confidence Threshold
    #     boxes, landms, scores = self.applyConfidenceThreshold(boxes, landms, scores) 
    #     detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32) 
        
    #     # Apply NMS
    #     keep = py_cpu_nms(detections, self.nms_thresh) 
    #     detections = detections[keep, :] 
    #     landms = landms[keep, :] 
    
    #     return detections, landms, scores

    def decodeBoxes(self, loc, conf, scale):
        """ Decode bounding boxes from model output. 
        Args:
            loc: Location predictions from the model.
            conf: Confidence predictions from the model.
            scale: Scale tensor for resizing boxes.
        Returns:
            boxes: Decoded bounding boxes.
            scores: Confidence scores.
        """

        boxes = decode(loc.data.squeeze(0), self.priors.data, self.cfg['variance'])
        boxes = boxes * scale.to(boxes.device, dtype=boxes.dtype)
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        return boxes, scores
    
    def decodeLandmarks(self, landms, img_width, img_height):
        """ Decode landmarks from model output.
        Args:
            landms: Landmark predictions from the model.
            img_width: Width of the original image.
            img_height: Height of the original image.
        Returns:
            landms: Decoded landmarks.
        """

        landms = decode_landm(landms.data.squeeze(0), self.priors.data, self.cfg['variance'])
        scale_landms = torch.tensor([img_width, img_height] * 5, device=landms.device, dtype=landms.dtype)
        print(f"Scale landmarks: {scale_landms}")
        landms = landms * scale_landms
        landms = landms.cpu().numpy()
        return landms
    
    def applyConfidenceThreshold(self, boxes, landms, scores):
        """ Apply confidence threshold to filter detections.
        Args:
            boxes: Detected bounding boxes.
            landms: Detected landmarks.
            scores: Confidence scores.
        Returns:
            boxes: Filtered bounding boxes.
            landms: Filtered landmarks.
            scores: Filtered confidence scores.
        """

        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        return boxes, landms, scores