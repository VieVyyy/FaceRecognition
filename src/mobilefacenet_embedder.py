import cv2
import numpy as np
from src.base_model import ModelLoader

class MobileFaceNetEmbedder(ModelLoader):
    """ MobileFaceNet Embedder using ONNX model """
    
    def __init__(self, onnx_path, device):
        super().__init__(onnx_path, device)

    def getEmbedding(self, image):
        """ Get face embedding from the input image.
        Args:
            image (numpy.ndarray): Input face image.
        Returns:
            embedding (numpy.ndarray): Face embedding vector.
        """

        image = cv2.resize(image, (112, 112))
        input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        input = np.expand_dims(input, axis = 0)
        input = input / 127.5 - 1.0     # normalize [-1, 1]

        embedding = self.infer(input)[0]
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
        embedding = embedding.flatten()

        return embedding
