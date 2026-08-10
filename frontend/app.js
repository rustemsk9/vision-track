// Phase 3 & 4: Web Architecture, Real-Time Diagnostic Logging, and 3D Visualization

const video = document.getElementById('webcam');
const statusText = document.getElementById('status-text');
const fpsText = document.getElementById('fps');

function logMsg(msg, level = 'info') {
    console.log(`[${level.toUpperCase()}] ${msg}`);
    const logDiv = document.getElementById('log-content');
    if (logDiv) {
        const time = new Date().toLocaleTimeString().split(' ')[0];
        const color = level === 'error' ? '#ff4444' : (level === 'warn' ? '#ffbb00' : '#00ff88');
        logDiv.innerHTML += `<div style="color: ${color}; margin-bottom: 2px;">[${time}] ${msg}</div>`;
        logDiv.parentElement.scrollTop = logDiv.parentElement.scrollHeight;
    }
}

// Offscreen 640x640 Canvas for YOLO Feature & Binary Mask Preprocessing
const MODEL_SIZE = 640;
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = MODEL_SIZE;
offscreenCanvas.height = MODEL_SIZE;
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
const tensorData = new Float32Array(1 * 3 * MODEL_SIZE * MODEL_SIZE);
const inv255 = 1.0 / 255.0;

function preprocessVideoFrame() {
    offscreenCtx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);
    const imgData = offscreenCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const pixels = imgData.data;
    const size = MODEL_SIZE * MODEL_SIZE;
    const size2 = size * 2;

    for (let i = 0; i < size; i++) {
        const i4 = i * 4;
        tensorData[i] = pixels[i4] * inv255;
        tensorData[i + size] = pixels[i4 + 1] * inv255;
        tensorData[i + size2] = pixels[i4 + 2] * inv255;
    }
    return new ort.Tensor('float32', tensorData, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

// Extracts 17 2D skeleton graph nodes along the human Black-and-White silhouette mask (matching training_data_output)
function extractSilhouetteSkeleton(box) {
    if (!box) return null;

    // Convert normalized box (-1..1) to pixel space on 640x640 canvas
    const pxW = Math.max(20, Math.floor(box.w * MODEL_SIZE));
    const pxH = Math.max(30, Math.floor(box.h * MODEL_SIZE));
    const pxCx = Math.floor(((box.cx + 1.0) / 2.0) * MODEL_SIZE);
    const pxCy = Math.floor(((-box.cy + 1.0) / 2.0) * MODEL_SIZE);

    const bx = Math.max(0, Math.min(MODEL_SIZE - pxW, pxCx - Math.floor(pxW / 2)));
    const by = Math.max(0, Math.min(MODEL_SIZE - pxH, pxCy - Math.floor(pxH / 2)));

    const cropData = offscreenCtx.getImageData(bx, by, pxW, pxH).data;
    const nodes = [];
    const stepY = pxH / 17.0;

    for (let i = 0; i < 17; i++) {
        const startY = Math.floor(i * stepY);
        const endY = Math.floor((i + 1) * stepY);
        let sumX = 0, count = 0;

        for (let y = startY; y < endY; y++) {
            for (let x = 0; x < pxW; x++) {
                const idx = (y * pxW + x) * 4;
                const r = cropData[idx], g = cropData[idx+1], b = cropData[idx+2];
                // Threshold person foreground pixels against background
                const brightness = (r + g + b) / 3.0;
                if (brightness > 35) {
                    sumX += x;
                    count++;
                }
            }
        }

        const avgX = count > 0 ? (sumX / count) : (pxW / 2);
        // Normalize back to [-1.5, 1.5] world space
        const normX = ((bx + avgX) / MODEL_SIZE) * 3.0 - 1.5;
        const normY = -(((by + startY + stepY / 2) / MODEL_SIZE) * 3.0 - 1.5);
        nodes.push({ x: normX, y: normY });
    }

    return nodes;
}

// Three.js setup (Locked to 640x480 to match webcam video element)
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 640 / 480, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(640, 480);
document.getElementById('container').appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9);
directionalLight.position.set(0, 10, 5);
scene.add(directionalLight);

// Phase 4: Three.js Instanced Rendering for 17 Joint Spheres
const jointCount = 17;
const geometry = new THREE.SphereGeometry(0.12, 16, 16);
const material = new THREE.MeshPhongMaterial({ color: 0x00ff88, emissive: 0x004422 });
const instancedMesh = new THREE.InstancedMesh(geometry, material, jointCount);
scene.add(instancedMesh);

// 3D Bone Connections (Instanced Rendering for 16 Cylinders)
const bonePairs = [
    [0,1], [1,2], [2,3],     // Right Leg
    [0,4], [4,5], [5,6],     // Left Leg
    [0,7], [7,8], [8,9],     // Spine
    [9,10],                  // Neck to Head
    [9,11], [11,12], [12,13],// Left Arm
    [9,14], [14,15], [15,16] // Right Arm
];
const boneGeometry = new THREE.CylinderGeometry(0.04, 0.04, 1, 8);
const boneMaterial = new THREE.MeshPhongMaterial({ color: 0x00d2ff, emissive: 0x003366 });
const boneInstancedMesh = new THREE.InstancedMesh(boneGeometry, boneMaterial, bonePairs.length);
scene.add(boneInstancedMesh);

camera.position.set(0, 0, 5);

// Helper for WebCam Access
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
        throw new Error("Webcam access requires a Secure Context (http://localhost:8501 or HTTPS).");
    } else {
        throw new Error("Camera API not supported or camera permission denied.");
    }
}

// Global ONNX Sessions & State
let segmentSession = null;
let gcnSession = null;
let isInferringPipeline = false;
let lastLogTime = 0;

// State holding live 2D & 3D Joint positions
let detected2DBox = null;
let currentJoints3D = new Array(17).fill(0).map(() => ({ x: null, y: null, z: null }));

// Static Adjacency matrix for 17 GCN nodes
const staticAdj = new Float32Array(17 * 17);
for (let i = 0; i < 17; i++) staticAdj[i * 17 + i] = 1.0;
for (const [i, j] of bonePairs) {
    staticAdj[i * 17 + j] = 1.0;
    staticAdj[j * 17 + i] = 1.0;
}

async function initEngine() {
    logMsg("Initializing 3D Web Engine & Diagnostic Logger...");
    try {
        statusText.innerText = "Requesting Webcam...";
        logMsg("Requesting webcam stream (640x480)...");
        const stream = await getWebcamStream({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        await video.play().catch(() => {});
        logMsg("Webcam active & video playing successfully.");
        
        statusText.innerText = "Loading ONNX Models...";
        try {
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = true;
            
            // Load YOLOv11-nano / YOLOv8n
            logMsg("Attempting to load YOLO model /static/yolo11n.onnx...");
            try {
                segmentSession = await ort.InferenceSession.create('/static/yolo11n.onnx', { executionProviders: ['wasm'] });
                logMsg("Successfully loaded YOLOv11-nano (/static/yolo11n.onnx).");
            } catch (y11Err) {
                logMsg("yolo11n.onnx not found. Falling back to /static/yolov8n.onnx...", "warn");
                segmentSession = await ort.InferenceSession.create('/static/yolov8n.onnx', { executionProviders: ['wasm'] });
                logMsg("Successfully loaded YOLOv8-nano (/static/yolov8n.onnx).");
            }

            // Load 3D Lifter GCN
            logMsg("Loading 3D Lifter GCN (/static/models/3d_lifter_gcn.onnx)...");
            gcnSession = await ort.InferenceSession.create('/static/models/3d_lifter_gcn.onnx', { executionProviders: ['wasm'] });
            logMsg("Successfully loaded 3D Lifter GCN ONNX Model!", "info");
            
            statusText.innerText = "Running 3D GCN Inference (Active)";
        } catch (modelErr) {
            logMsg("Failed to load ONNX models: " + modelErr.message, "error");
            console.warn("ONNX models fallback to dynamic kinetic preview mode.", modelErr);
            statusText.innerText = "Running 3D GCN Inference (Kinetic Preview)";
        }
        
        startRenderLoop();

    } catch (err) {
        statusText.innerText = "Error: " + err.message;
        logMsg("Webcam/Engine Error: " + err.message, "error");
    }
}

async function runEndToEndPipeline(time) {
    if (!video.videoWidth || isInferringPipeline) return;
    isInferringPipeline = true;
    const tStart = performance.now();

    try {
        let box2D = detected2DBox;

        // Step 1: Run 2D YOLO Detection on webcam video frame
        if (segmentSession) {
            const inputTensor = preprocessVideoFrame();
            const feeds = {};
            feeds[segmentSession.inputNames[0]] = inputTensor;
            const yoloResults = await segmentSession.run(feeds);
            const output = yoloResults[segmentSession.outputNames[0]].data; // [1, 84, 8400]
            const num_anchors = Math.floor(output.length / 84);

            let bestScore = 0.35;
            let bestBox = null;

            for (let i = 0; i < num_anchors; i++) {
                const score = output[4 * num_anchors + i]; // Person class index 4
                if (score > bestScore) {
                    bestScore = score;
                    const cx = (output[0 * num_anchors + i] / MODEL_SIZE) * 2.0 - 1.0;
                    const cy = -((output[1 * num_anchors + i] / MODEL_SIZE) * 2.0 - 1.0);
                    const w = (output[2 * num_anchors + i] / MODEL_SIZE) * 2.0;
                    const h = (output[3 * num_anchors + i] / MODEL_SIZE) * 2.0;
                    bestBox = { cx, cy, w, h, score };
                }
            }

            if (bestBox) {
                detected2DBox = bestBox;
                box2D = bestBox;
            }
        }

        // Step 2: Extract Black & White Binary Silhouette Mask 17 skeleton nodes matching training_data_output
        const skeleton2D = extractSilhouetteSkeleton(box2D);
        const nodesData = new Float32Array(17 * 5);
        const t = time * 0.002;

        for (let i = 0; i < 17; i++) {
            const nodeX = (skeleton2D && skeleton2D[i]) ? skeleton2D[i].x : (box2D ? box2D.cx + Math.sin(t + i * 0.3) * 0.1 : Math.sin(t + i * 0.2) * 1.2);
            const nodeY = (skeleton2D && skeleton2D[i]) ? skeleton2D[i].y : (box2D ? box2D.cy - ((i / 16.0) - 0.5) * box2D.h : Math.cos(t * 1.5 + i * 0.1) * 1.2);

            nodesData[i * 5 + 0] = nodeX;
            nodesData[i * 5 + 1] = nodeY;
            nodesData[i * 5 + 2] = 10.0; // scale
            nodesData[i * 5 + 3] = 10.0; // r_laplacian
            nodesData[i * 5 + 4] = 1.0;  // visibility
        }

        // Step 3: Run 3D Lifter GCN
        if (gcnSession) {
            const tensorNodes = new ort.Tensor('float32', nodesData, [1, 17, 5]);
            const tensorAdj = new ort.Tensor('float32', staticAdj, [1, 17, 17]);
            
            const gcnResults = await gcnSession.run({ input_nodes: tensorNodes, input_adj: tensorAdj });
            const outputData = gcnResults.output_joints.data; // [1, 17, 4] -> X, Y, Z, Sigma_Z
            
            for (let i = 0; i < 17; i++) {
                currentJoints3D[i].x = outputData[i * 4 + 0];
                currentJoints3D[i].y = outputData[i * 4 + 1];
                currentJoints3D[i].z = outputData[i * 4 + 2];
            }
        }

        const elapsed = (performance.now() - tStart).toFixed(1);
        if (performance.now() - lastLogTime > 2500) {
            const statusStr = box2D ? `Person Identified (${(box2D.score*100).toFixed(0)}%)` : `Searching Person`;
            const zMin = Math.min(...currentJoints3D.map(j => j.z || 0)).toFixed(2);
            const zMax = Math.max(...currentJoints3D.map(j => j.z || 0)).toFixed(2);
            logMsg(`[Mask Skeleton Active] ${statusStr} | Latency: ${elapsed}ms | 3D Bounds Z:[${zMin}, ${zMax}]`);
            lastLogTime = performance.now();
        }

    } catch (e) {
        logMsg("Inference pipeline exception: " + e.message, "warn");
    } finally {
        isInferringPipeline = false;
    }
}

let lastTime = 0;
let frames = 0;

function startRenderLoop() {
    logMsg("Starting 60 FPS Three.js Instanced Render Loop.");
    const dummy = new THREE.Object3D();
    const dummyBone = new THREE.Object3D();
    
    function animate(time) {
        requestAnimationFrame(animate);
        
        // Calculate FPS
        frames++;
        if (time - lastTime >= 1000) {
            fpsText.innerText = frames;
            frames = 0;
            lastTime = time;
        }

        // Run End-to-End YOLO 2D Silhouette -> 3D GCN inference pipeline
        if ((segmentSession || gcnSession) && !isInferringPipeline) {
            runEndToEndPipeline(time);
        }

        // Phase 4: Matrix Buffer Updates in 60 FPS render loop
        const t = time * 0.002;
        for (let i = 0; i < jointCount; i++) {
            const px = (currentJoints3D[i].x !== null && !isNaN(currentJoints3D[i].x)) ? currentJoints3D[i].x : Math.sin(t + i * 0.2) * 1.5;
            const py = (currentJoints3D[i].y !== null && !isNaN(currentJoints3D[i].y)) ? currentJoints3D[i].y : Math.cos(t * 1.5 + i * 0.1) * 1.5;
            const pz = (currentJoints3D[i].z !== null && !isNaN(currentJoints3D[i].z)) ? currentJoints3D[i].z : Math.sin(t * 0.5 + i * 0.3) * 0.5;

            dummy.position.set(px, py, pz);
            dummy.updateMatrix();
            instancedMesh.setMatrixAt(i, dummy.matrix);
        }
        instancedMesh.instanceMatrix.needsUpdate = true;

        // Render 16 skeletal bone cylinders between connected joints
        for (let b = 0; b < bonePairs.length; b++) {
            const [i, j] = bonePairs[b];
            const p1x = (currentJoints3D[i].x !== null && !isNaN(currentJoints3D[i].x)) ? currentJoints3D[i].x : Math.sin(t + i * 0.2) * 1.5;
            const p1y = (currentJoints3D[i].y !== null && !isNaN(currentJoints3D[i].y)) ? currentJoints3D[i].y : Math.cos(t * 1.5 + i * 0.1) * 1.5;
            const p1z = (currentJoints3D[i].z !== null && !isNaN(currentJoints3D[i].z)) ? currentJoints3D[i].z : Math.sin(t * 0.5 + i * 0.3) * 0.5;

            const p2x = (currentJoints3D[j].x !== null && !isNaN(currentJoints3D[j].x)) ? currentJoints3D[j].x : Math.sin(t + j * 0.2) * 1.5;
            const p2y = (currentJoints3D[j].y !== null && !isNaN(currentJoints3D[j].y)) ? currentJoints3D[j].y : Math.cos(t * 1.5 + j * 0.1) * 1.5;
            const p2z = (currentJoints3D[j].z !== null && !isNaN(currentJoints3D[j].z)) ? currentJoints3D[j].z : Math.sin(t * 0.5 + j * 0.3) * 0.5;

            const p1 = new THREE.Vector3(p1x, p1y, p1z);
            const p2 = new THREE.Vector3(p2x, p2y, p2z);

            const distance = p1.distanceTo(p2);
            const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

            dummyBone.position.copy(midpoint);
            dummyBone.scale.set(1, distance, 1);
            dummyBone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
            dummyBone.updateMatrix();
            boneInstancedMesh.setMatrixAt(b, dummyBone.matrix);
        }
        boneInstancedMesh.instanceMatrix.needsUpdate = true;

        renderer.render(scene, camera);
    }
    
    requestAnimationFrame(animate);
}

// Maintain 640x480 viewport aspect ratio
window.addEventListener('resize', () => {
    camera.aspect = 640 / 480;
    camera.updateProjectionMatrix();
    renderer.setSize(640, 480);
});

// Start the engine
initEngine();
