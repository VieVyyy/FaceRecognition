# Face Recognition Web Demo with Flask & SocketIO

This project provides a real-time face recognition system via a web interface. It utilizes [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface.git) for detection, [MobileFaceNet](https://github.com/foamliu/MobileFaceNet.git) for feature extraction, and FAISS for high-speed vector searching.

## Key Features

- **Real-time Recognition:** Live streaming from the webcam to the browser using Flask-SocketIO.

- **Face Registration:** Add and register new faces into the database directly through the web UI.

- **High Performance:** Uses a FAISS Index to search 128-dimensional embeddings with extremely low latency.

- **Hardware Optimization:** Supports GPU acceleration via CUDA for faster inference.

## Tech Stack

- **Backend:** Flask, Flask-SocketIO (WebSockets).

- **Computer Vision:** OpenCV.

- **Deep Learning:** PyTorch (RetinaFace and MobileFaceNet models).

- **Vector Database:** FAISS (Facebook AI Similarity Search).

## System Requirements

- Python 3.8+

- A functional Webcam.

- (Optional) NVIDIA GPU + CUDA for optimal performance.

**Required Libraries:**

```bash
pip install flask flask-socketio opencv-python torch numpy faiss-cpu
```

## Installation and Setup

### 1. Prepare Models

- Ensure the following model files are placed in the models/ directory:

- `RetinaFaceNet.onnx` (Detector)

- `MobileFaceNet.onnx` (Embedder)

### 2. Install Libraries

Create a virtual environment (recommended) and install the dependencies:

```bash
# Install required libraries
pip install -r requirements.txt
```

### 3. Run the Application

Start the Flask server by running:

```bash
python main.py
```

### 4. Access the Web Interface

Open your browser and navigate to:

- **Recognition:** `http://127.0.0.1:5000/`

- **Add New Face:** `http://127.0.0.1:5000/add`

## How it Works (main.py)

- `streamVideo`: A background task that reads frames from the camera, performs face detection/alignment, and searches the FAISS index. Results are drawn on the frame and emitted to the client via the video_frame socket.

- `addFaceToIndex`: Decodes base64 images from the client, aligns the face, extracts a 128-dimensional embedding, and saves it to the FAISS index and local storage.

- `Thresholding`: The THRESHOLD variable (default 1.1 - 1.3) determines the distance at which a face is classified as "Unknown".
