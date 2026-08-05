const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');

// Offscreen canvas for YOLO 640x640 tensor resizing
// willReadFrequently=true tells the browser to keep pixel data in CPU-accessible RAM,
// preventing the GPU→CPU readback penalty that Chrome warns about on every getImageData() call
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 640;
offscreenCanvas.height = 640;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

let session;

async function getWebcamStream(constraints = { video: { width: 640, height: 480 } }) {
    if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function') {
        return await navigator.mediaDevices.getUserMedia(constraints);
    }
    
    const legacyGetUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
    if (legacyGetUserMedia) {
        return new Promise((resolve, reject) => {
            legacyGetUserMedia.call(navigator, constraints, resolve, reject);
        });
    }

    if (!window.isSecureContext) {
        throw new Error("Webcam access requires a Secure Context (e.g. http://localhost:8501 or HTTPS). Modern browsers block camera access on insecure origins like http://" + window.location.hostname);
    } else {
        throw new Error("Camera API (navigator.mediaDevices.getUserMedia) is not supported by your browser or camera permission was denied.");
    }
}

async function initWebcam() {
    try {
        statusDiv.innerText = "Requesting Webcam Permission...";
        const stream = await getWebcamStream({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        video.play().catch(() => {});
        
        video.onloadedmetadata = async () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            statusDiv.innerText = "Downloading YOLO Model (12MB). Please wait...";
            
            // Disable threading to prevent Safari from requesting SharedArrayBuffer and crashing on threaded WASM
            ort.env.wasm.numThreads = 1;
            
            // Explicitly set the WASM paths because Safari struggles to resolve relative WASM blobs from about:srcdoc iframes
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
            
            // Fetch model from Streamlit's static server using same-origin relative URL.
            // Fetch model from the Go Gateway static server.
            // This avoids Streamlit's macOS MIME type serving bugs completely.
            const modelUrl = '/static/yolov8n.onnx';
            
            // Explicitly fetch the model as an ArrayBuffer to bypass Safari's Range Request
            // truncation bug on large binary files. Safari sometimes sends Range: bytes=0-
            // headers and then silently drops the connection mid-download, causing ONNX to
            // receive a truncated payload and crash with "protobuf parsing failed".
            const response = await fetch(modelUrl, { cache: 'force-cache' });
            const arrayBuffer = await response.arrayBuffer();
            
            // Safety check: the ONNX model is ~12MB. If we got less than 10KB, Safari truncated it.
            if (arrayBuffer.byteLength < 10000) {
                throw new Error("Model file too small (" + arrayBuffer.byteLength + " bytes). The download was likely truncated by the browser.");
            }
            
            console.log("[Edge AI] Model downloaded successfully: " + (arrayBuffer.byteLength / 1024 / 1024).toFixed(1) + " MB");
            
            const modelBytes = new Uint8Array(arrayBuffer);
            
            // Try WebGL first for GPU acceleration (~30+ FPS), fallback to WASM CPU
            session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['webgl', 'wasm'] });
            
            statusDiv.innerText = "Edge AI Active! (Running at ~30 FPS)";
            setTimeout(() => { statusDiv.style.display = 'none'; }, 3000);
            
            inferenceLoop();
        };
    } catch (e) {
        statusDiv.innerText = "Error: " + e.message;
        console.error(e);
    }
}

// Pre-allocate typed array memory once to eliminate Garbage Collection pauses
const tensorData = new Float32Array(1 * 3 * 640 * 640);
const inv255 = 1.0 / 255.0;

// Convert HTML5 ImageData to Float32 CHW Tensor (1, 3, 640, 640)
function preprocess(imageBuffer) {
    const size = 640 * 640;
    const size2 = size * 2;
    for (let i = 0; i < size; i++) {
        const i4 = i * 4;
        tensorData[i] = imageBuffer[i4] * inv255;
        tensorData[i + size] = imageBuffer[i4 + 1] * inv255;
        tensorData[i + size2] = imageBuffer[i4 + 2] * inv255;
    }
    return new ort.Tensor('float32', tensorData, [1, 3, 640, 640]);
}

// Compute Intersection Over Union for NMS
function computeIOU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);
    
    if (x2 < x1 || y2 < y1) return 0.0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    
    return intersection / (area1 + area2 - intersection);
}

let fpsCounter = 0;
let currentFps = 0;
let currentLatency = 0;
let fpsTimer = performance.now();

async function inferenceLoop() {
    if (!video.videoWidth) {
        setTimeout(inferenceLoop, 100);
        return;
    }

    const tStart = performance.now();

    // 1. Draw video frame to 640x640 offscreen canvas
    offscreenCtx.drawImage(video, 0, 0, 640, 640);
    const imgData = offscreenCtx.getImageData(0, 0, 640, 640);
    
    // 2. Preprocess to tensor
    const input = preprocess(imgData.data);
    
    // 3. Run Inference directly inside the browser sandbox!
    const feeds = {};
    feeds[session.inputNames[0]] = input;
    const results = await session.run(feeds);
    
    const output = results[session.outputNames[0]].data; // [1, 84, 8400]
    
    // 4. Post-process (Filter Person Class = index 4)
    let boxes = [];
    for (let i = 0; i < 8400; i++) {
        let person_score = output[4 * 8400 + i];
        if (person_score > 0.45) { // Confidence Threshold
            let cx = output[0 * 8400 + i];
            let cy = output[1 * 8400 + i];
            let w = output[2 * 8400 + i];
            let h = output[3 * 8400 + i];
            boxes.push({
                x1: cx - w / 2,
                y1: cy - h / 2,
                x2: cx + w / 2,
                y2: cy + h / 2,
                score: person_score
            });
        }
    }
    
    // Non-Maximum Suppression (NMS)
    boxes.sort((a, b) => b.score - a.score);
    let finalBoxes = [];
    for (let b of boxes) {
        let keep = true;
        for (let f of finalBoxes) {
            if (computeIOU(b, f) > 0.45) {
                keep = false;
                break;
            }
        }
        if (keep) finalBoxes.push(b);
    }
    
    const tEnd = performance.now();
    currentLatency = tEnd - tStart;

    // Calculate real-time FPS
    fpsCounter++;
    if (tEnd - fpsTimer >= 1000) {
        currentFps = Math.round((fpsCounter * 1000) / (tEnd - fpsTimer));
        fpsCounter = 0;
        fpsTimer = tEnd;
    }
    
    // 5. Draw Results
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = canvas.width / 640.0;
    const scaleY = canvas.height / 640.0;
    
    for (let b of finalBoxes) {
        const px1 = b.x1 * scaleX;
        const py1 = b.y1 * scaleY;
        const pw = (b.x2 - b.x1) * scaleX;
        const ph = (b.y2 - b.y1) * scaleY;
        
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 4;
        ctx.strokeRect(px1, py1, pw, ph);
        
        ctx.fillStyle = '#00FF00';
        ctx.font = '20px Arial';
        ctx.fillText("Person: " + (b.score * 100).toFixed(0) + "%", px1, py1 - 10);
    }
    
    ctx.fillStyle = '#00FF00';
    ctx.font = 'bold 22px Arial';
    ctx.fillText(`⚡ Edge AI: ${currentFps} FPS | Latency: ${currentLatency.toFixed(1)} ms`, 20, 35);
    
    ctx.fillStyle = '#FF0000';
    ctx.font = '22px Arial';
    ctx.fillText("Total People: " + finalBoxes.length, 20, 65);
    
    // 6. Schedule next frame immediately (no 33ms delay throttle)
    setTimeout(inferenceLoop, 0);
}

// Start
initWebcam();
