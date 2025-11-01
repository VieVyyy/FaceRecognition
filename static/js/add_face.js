const socket = io();

const video = document.getElementById('cameraFeed');
const canvas = document.getElementById('captureCanvas');
const addFaceButton = document.getElementById('addFaceButton');
const labelInput = document.getElementById('faceLabel');
const statusMessage = document.getElementById('statusMessage');

// Khỏi động camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
    } catch (error) {
        alert('Không thể truy cập camera: ' + error.message);
        statusMessage.color = 'red';
        statusMessage.innerText = 'Lỗi: Không thể truy cập camera.';
        addFaceButton.disabled = true;
    }
}

startCamera();

// Chụp ảnh và gửi lên server để thêm khuôn mặt
addFaceButton.addEventListener('click', function() {
    const label = labelInput.value.trim();

    if (label === '') {
        alert('Vui lòng nhập nhãn cho khuôn mặt.');
        return;
    }

    // Vẽ khung hình hiện tại lên canvas
    const context = canvas.getContext('2d');

    // Lật ảnh ngang để phù hợp với hình ảnh camera
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Lấy dữ liệu ảnh từ canvas
    const imageDataURL = canvas.toDataURL('image/jpeg');
    const base64Image = imageDataURL.split(',')[1];

    // Gửi ảnh và nhãn lên server
    addFaceButton.disabled = true;
    statusMessage.color = 'pink';
    statusMessage.innerText = 'Đang thêm khuôn mặt...';
    socket.emit('add_face', { image: base64Image, label: label });
});

socket.on('add_face_response', function(data) {
    if (data.success) {
        statusMessage.color = 'green';
        statusMessage.innerText = 'Khuôn mặt đã được thêm thành công!';
    } else {
        statusMessage.color = 'red';
        statusMessage.innerText = 'Lỗi khi thêm khuôn mặt: ' + data.message;
    }
    addFaceButton.disabled = false;
});