import os
import cv2
import torch
import numpy as np
from src.face_recognizer import FaceRecognizer
from src.faiss_index import FaceIndex
from utils.image_utils import *

from flask import Flask, render_template
from flask_socketio import SocketIO
import base64

# Khai báo ứng dụng Flask và SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_INDEX_PATH = "face_data/emb_face/face_index.bin"
THRESHOLD = 1.3  # Ngưỡng khoảng cách để xác định Unknown (càng nhỏ càng chặt)
is_streaming = False # Biến trạng thái luồng video

detector_model = "models/RetinaFaceNet.onnx"
embedder_model = "models/MobileFaceNet.onnx"

# Khởi tạo nhận diện và index
recognizer = FaceRecognizer(detector_model, embedder_model, device=DEVICE)
faiss_index = FaceIndex(dim=128, index_path=FACE_INDEX_PATH)

org_img_dir = "face_data/org_face/"
if not os.path.exists(FACE_INDEX_PATH):
    print("Tạo FAISS index từ ảnh gốc...")
    faiss_index.create(org_img_dir, recognizer)

print("Face recognizer and FAISS index loaded.")

def encodeFrameToBase64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

@socketio.on('start_stream')
def handleStartStream():
    global is_streaming
    if not is_streaming:
        print("Client kết nối real-time face recognition.")
        is_streaming = True
        socketio.start_background_task(target=streamVideo)
    else:
        print("Stream đang chạy!")

@socketio.on('stop_stream')
def handleStopStream():
    global is_streaming
    if is_streaming:
        print("Client ngắt kết nối real-time face recognition.")
        is_streaming = False
    else:
        print("Stream chưa chạy!")

def streamVideo(threshold=1.3):
    global is_streaming 

    cap = cv2.VideoCapture(0)
    frame_id = 0
    if not cap.isOpened():
        print("Không thể mở camera!")
        is_streaming = False
        socketio.emit('stream_error', {'message': 'Không thể mở camera!'})
        return
    
    try:
        while is_streaming:
            success, frame = cap.read()
            if not success:
                print("Không đọc được khung hình!")
                break

            frame_id += 1
            aligned_faces, bboxes, _, _ = recognizer.detectAndAlign(frame)
            results = []

            for face, bbox in zip(aligned_faces, bboxes):
                emb = recognizer.getEmbeddingFromAligned(face)
                if emb is not None:
                    labels, distances = faiss_index.search(emb, top_k=1)
                    label = labels[0]
                    distance = distances[0]

                    if distance > threshold:
                        label = "Unknown"

                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} ({distance:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            frame_base64 = encodeFrameToBase64(frame)
            socketio.emit('video_frame', {'image': frame_base64})
            socketio.sleep(0.07)  # Giới hạn tốc độ khung hình

            if frame_id % 15 == 0:
                del frame, aligned_faces, bboxes, results
                torch.cuda.empty_cache() # Giải phóng bộ nhớ GPU định kỳ

    finally:
        print("Dừng stream video.")
        cap.release()
        is_streaming = False
        socketio.emit('stream_stopped', {'message': 'Stream đã dừng.'})

@app.route('/')
def index():
    return render_template('index.html')
        
if __name__ == '__main__':
    socketio.run(app, debug=False)