const socket = io();

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const img = document.getElementById('video');


function startStream() {
    socket.emit('start_stream');
    startButton.disabled = true;
    stopButton.disabled = false;
}

function stopStream() {
    socket.emit('stop_stream');
    startButton.disabled = false;
    stopButton.disabled = true;
    img.src = '';
}

socket.on('video_frame', function(data) {
    img.src = 'data:image/jpeg;base64,' + data.image;
});

socket.on('stream_error', function(data) {
    alert(data.message);
    startButton.disabled = false;
    stopButton.disabled = true;
});

socket.on('stream_stopped', function() {
    console.log("Stream đã dừng.");
    startButton.disabled = false;
    stopButton.disabled = true;
});