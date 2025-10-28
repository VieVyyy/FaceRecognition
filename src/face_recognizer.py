import torch
import cv2
import numpy as np

from utils.image_utils import *
from src.retinaface_detector import RetinaFaceDetector
from src.mobilefacenet_embedder import MobileFaceNetEmbedder

class FaceRecognizer:
    def __init__(self, detector_model_path, embedder_model_path, device='cpu'):
        self.device = torch.device(device)
        self.detector = RetinaFaceDetector(detector_model_path, device)
        self.embedder = MobileFaceNetEmbedder(embedder_model_path, device)

    def detectAndAlign(self, image_input):
        """ Detect and align faces in the input image.
        Args:
            image_path (path or array): Path to the input image.
        Returns:
            aligned_faces (list): List of aligned face images.
            detections (numpy.ndarray): Detected bounding boxes.
            landms (numpy.ndarray): Detected landmarks.
            scores (numpy.ndarray): Confidence scores.
        """
        image = None
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("image_input phải là đường dẫn hoặc numpy array")

        if image is None:
            raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ!")

        detections, landms, scores = self.detector.detect(image)
        if detections is None or len(detections) == 0:
            print("⚠️ Không phát hiện được khuôn mặt nào.")
            return [], [], [], []
        
        aligned_faces = []
        bboxes = []
        for i, det in enumerate(detections):
            try:
                aligned_face = alignFace(image, landms[i], det)
                aligned_faces.append(aligned_face)
                bboxes.append(det)
            except Exception as e:
                print(f"⚠️ Không thể căn chỉnh khuôn mặt {i}: {e}")


        return aligned_faces, detections, landms, scores
    
    def getEmbeddings(self, image_path):
        """ Get face embeddings from the input image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            embeddings (list): List of face embedding vectors.
        """

        aligned_faces, _, _, _ = self.detectAndAlign(image_path)

        embeddings = []
        for face in aligned_faces:
            embedding = self.embedder.getEmbedding(face)
            embeddings.append(embedding)

        return embeddings

    def getEmbeddingFromAligned(self, aligned_face):
        """ Trích xuất embedding từ ảnh khuôn mặt đã căn chỉnh (cho real-time). """
        if aligned_face is None:
            raise ValueError("Ảnh đầu vào rỗng trong getEmbeddingFromAligned")

        # Resize về kích thước phù hợp với model (MobileFaceNet: 112x112)
        face = cv2.resize(aligned_face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Chuẩn hóa về [-1, 1]
        face = (face / 127.5) - 1.0
        face = np.transpose(face, (2, 0, 1))  # (C,H,W)
        face = np.expand_dims(face, axis=0)

        # Trích xuất embedding
        embedding = self.embedder.infer(face)
        embedding = np.array(embedding).squeeze()

        # Chuẩn hóa vector L2
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)