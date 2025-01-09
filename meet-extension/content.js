// content.js
let isEnabled = false;
let ws = null;
let faceId = null;
let frameProcessor = null;

class FrameProcessor {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 640;
        this.canvas.height = 480;
        this.processingFrame = false;
    }

    async processVideoFrame(videoElement) {
        if (this.processingFrame) return;
        this.processingFrame = true;

        try {
            // Create a VideoFrame
            const frame = new VideoFrame(videoElement);
            
            // Draw the frame to canvas
            this.canvas.width = frame.displayWidth;
            this.canvas.height = frame.displayHeight;
            this.ctx.drawImage(videoElement, 0, 0, this.canvas.width, this.canvas.height);
            
            // Always close the frame after using it
            frame.close();

            // Convert to base64
            const dataUrl = this.canvas.toDataURL('image/jpeg', 0.8);
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    face_id: faceId,
                    frame: dataUrl,
                    timestamp: Date.now()
                }));
            }
        } catch (error) {
            console.error('Frame processing error:', error);
        } finally {
            this.processingFrame = false;
        }
    }

    destroy() {
        this.ctx = null;
        this.canvas = null;
    }
}

function setupVideoProcessing(videoElement) {
    if (!videoElement) return;

    // Create overlay canvas
    const overlay = document.createElement('canvas');
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.zIndex = '1';

    // Add canvas to video container
    const container = videoElement.parentElement;
    if (container) {
        container.style.position = 'relative';
        container.appendChild(overlay);
    }

    const ctx = overlay.getContext('2d');
    frameProcessor = new FrameProcessor();

    // Handle WebSocket messages
    function handleWsMessage(event) {
        const data = JSON.parse(event.data);
        if (data.frame) {
            const img = new Image();
            img.onload = () => {
                overlay.width = videoElement.videoWidth;
                overlay.height = videoElement.videoHeight;
                ctx.drawImage(img, 0, 0, overlay.width, overlay.height);
            };
            img.src = data.frame;
        }
    }

    // Process frames
    async function processFrames() {
        if (!isEnabled || !videoElement.videoWidth) return;

        await frameProcessor.processVideoFrame(videoElement);
        requestAnimationFrame(processFrames);
    }

    // Set up WebSocket
    function connectWebSocket() {
        ws = new WebSocket('ws://localhost:8000/ws');
        ws.onopen = () => {
            console.log('WebSocket connected');
            requestAnimationFrame(processFrames);
        };
        ws.onmessage = handleWsMessage;
        ws.onclose = () => {
            console.log('WebSocket closed');
            if (isEnabled) {
                setTimeout(connectWebSocket, 1000);
            }
        };
    }

    if (isEnabled) {
        connectWebSocket();
    }

    return () => {
        isEnabled = false;
        if (ws) ws.close();
        if (frameProcessor) frameProcessor.destroy();
        overlay.remove();
    };
}

// Find and setup video elements
function setupVideo() {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
        if (video.srcObject && !video.__faceSwapInitialized) {
            video.__faceSwapInitialized = true;
            setupVideoProcessing(video);
        }
    });
}

// Handle extension messages
chrome.runtime.onMessage.addListener(async (message) => {
    if (message.type === 'TOGGLE_SWAP') {
        isEnabled = message.enabled;
        
        if (isEnabled) {
            const result = await chrome.storage.local.get(['face_id']);
            if (result.face_id) {
                faceId = result.face_id;
                setupVideo();
            }
        } else {
            if (ws) ws.close();
            if (frameProcessor) frameProcessor.destroy();
        }
    }
});

// Monitor for new video elements
const observer = new MutationObserver(() => {
    if (isEnabled) setupVideo();
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});