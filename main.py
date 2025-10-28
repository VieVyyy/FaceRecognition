# import torch
# import cv2
# from utils.image_utils import *
# from src.retinaface_detector import RetinaFaceDetector
# from src.face_recognizer import FaceRecognizer
# from src.faiss_index import FaceIndex
# import os
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# detector_model = "models/RetinaFaceNet.onnx"
# embedder_model = "models/MobileFaceNet.onnx"
# IMG_SIZE = (640, 640)
# detector = RetinaFaceDetector(detector_model, device, input_size=IMG_SIZE)

# recognizer = FaceRecognizer(detector_model, embedder_model, device)
# faiss_index = FaceIndex(dim=128, index_path="face_data/emb_face/face_index.bin")

# # # --- Thêm khuôn mặt mới ---
# # embeddings = recognizer.getEmbeddings("face_data/org_face/vy_img.jpg")
# # for emb in embeddings:
# #     faiss_index.add(emb, "Vy")
# # faiss_index.saveIndex()

# # --- Nhận diện khuôn mặt mới ---
# test_embeddings = recognizer.getEmbeddings("face_data/test_images/VY/vy_1.jpg")
# for emb in test_embeddings:
#     labels, dists = faiss_index.search(emb)
#     print(f"Nhận diện: {labels[0]} (distance={dists[0]:.4f})")

# test_img_dir = "face_data/test_images/"
# count = 0
# for folder in os.listdir(test_img_dir):
#     folder_path = os.path.join(test_img_dir, folder)
#     if os.path.isdir(folder_path):
#         for filename in os.listdir(folder_path):
#             if filename.endswith(".jpg") or filename.endswith(".png"):
#                 img_path = os.path.join(folder_path, filename)
#                 embeddings = recognizer.getEmbeddings(img_path)
#                 for emb in embeddings:
#                     labels, dists = faiss_index.search(emb)
#                     print(f"Ảnh: {filename} - Nhận diện: {labels[0]} (distance={dists[0]:.4f})")
#                     if labels[0] == folder.lower():
#                         count += 1
# print(f"Accuracy: {count}/{sum(len(os.listdir(os.path.join(test_img_dir, f))) for f in os.listdir(test_img_dir) if os.path.isdir(os.path.join(test_img_dir, f)))} = {count / sum(len(os.listdir(os.path.join(test_img_dir, f))) for f in os.listdir(test_img_dir) if os.path.isdir(os.path.join(test_img_dir, f))):.4f}")

import os
import cv2
import torch
import numpy as np
from src.face_recognizer import FaceRecognizer
from src.faiss_index import FaceIndex
from utils.image_utils import *

# ==============================
# Cấu hình
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_INDEX_PATH = "face_data/emb_face/face_index.bin"
# THRESHOLD = 1.4  # Ngưỡng khoảng cách để xác định Unknown (càng nhỏ càng chặt)

detector_model = "models/RetinaFaceNet.onnx"
embedder_model = "models/MobileFaceNet.onnx"

# ==============================
# Khởi tạo nhận diện và index
# ==============================
recognizer = FaceRecognizer(detector_model, embedder_model, device=DEVICE)
faiss_index = FaceIndex(dim=128, index_path=FACE_INDEX_PATH)

org_img_dir = "face_data/org_face/"
if not os.path.exists(FACE_INDEX_PATH):
    print("Tạo FAISS index từ ảnh gốc...")
    faiss_index.create(org_img_dir, recognizer)

print("Face recognizer and FAISS index loaded.")
THRESHOLD = faiss_index.suggestOptimalThreshold()
# ==============================



# Bắt đầu camera
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được khung hình!")
        break

    # Phát hiện khuôn mặt và trích xuất embeddings
    try:
        aligned_faces, bboxes, landms, _ = recognizer.detectAndAlign(frame)

        for face, bbox, landm in zip(aligned_faces, bboxes, landms):
            emb = recognizer.getEmbeddingFromAligned(face)
            if emb is not None:
                # Tìm kiếm trong FAISS index
                labels, distances = faiss_index.search(emb, top_k=1)
                label = labels[0]
                distance = distances[0]

                if distance > THRESHOLD:
                    label = "Unknown"

                # Vẽ khung và nhãn
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} ({distance:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    except Exception as e:
        print("⚠️ Lỗi phát hiện:", e)

    # Hiển thị khung hình
    cv2.imshow("Real-time Face Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
