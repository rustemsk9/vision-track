// Phase 3 & 4: Web Architecture, 2D YOLO Pose Keypoints & Dual Engine (ONNX GCN / 3D Kinematic)

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

// Global Engine Selector: 'kinematic' or 'gcn_onnx'
let activeEngineMode = 'kinematic';

function setEngineMode(mode) {
    activeEngineMode = mode;
    logMsg(`Switched 3D Engine Mode to: ${mode === 'gcn_onnx' ? '3D Lifter GCN (ONNX)' : '3D Kinematic Engine'}`);
    const modeBadge = document.getElementById('mode-badge');
    if (modeBadge) {
        modeBadge.innerText = mode === 'gcn_onnx' ? '3D Lifter GCN (ONNX)' : '3D Kinematic Engine';
        modeBadge.style.color = mode === 'gcn_onnx' ? '#ffbb00' : '#00ff88';
    }

    if (typeof skeletonGroup !== 'undefined' && skeletonGroup) {
        skeletonGroup.scale.set(1.0, 1.0, 1.0);
        camera.position.set(0, 0, 3.8);
    }
}

// Offscreen 640x640 Canvas for YOLO Preprocessing
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

// Three.js setup (Locked to 640x480 to match webcam video element)
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 640 / 480, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(640, 480);
document.getElementById('container').appendChild(renderer.domElement);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
directionalLight.position.set(0, 10, 5);
scene.add(directionalLight);

// Phase 4: Three.js Instanced Rendering for 17 Joint Spheres
const jointCount = 17;
const geometry = new THREE.SphereGeometry(0.12, 16, 16);
const material = new THREE.MeshPhongMaterial({ color: 0x00ff88, emissive: 0x004422 });
const instancedMesh = new THREE.InstancedMesh(geometry, material, jointCount);

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

// Create skeleton parent container for runtime scene scaling
const skeletonGroup = new THREE.Group();
scene.add(skeletonGroup);
skeletonGroup.add(instancedMesh);
skeletonGroup.add(boneInstancedMesh);

// Fix 1 Option A: Boost display height by 25% via skeletonGroup container scale
skeletonGroup.scale.set(1.25, 1.25, 1.25);

// Fix 1 Option B: Move camera closer (Z=3.2) so subject fills canvas view
camera.position.set(0, 0, 3.2);

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
let poseSession = null;
let gcnSession = null;
let isInferringPipeline = false;
let lastLogTime = 0;

// World space tracking state
let personCenterX = 0.5;
let personCenterY = 0.5;

// State holding live 2D & 3D Joint positions
let currentJoints3D = new Array(17).fill(0).map(() => ({ x: 0, y: 0, z: 0 }));
let keypoints2D = new Array(17).fill(0).map(() => ({ x: null, y: null, conf: 0 }));

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
            ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
            ort.env.wasm.simd = true;
            const providers = ['webgpu', 'webgl', 'wasm'];

            logMsg(`Loading 2D Pose Model (/static/yolov8n-pose.onnx) with ${ort.env.wasm.numThreads} WASM threads...`);
            try {
                poseSession = await ort.InferenceSession.create('/static/yolov8n-pose.onnx', { executionProviders: providers });
                logMsg("Successfully loaded YOLOv8-Pose 2D Keypoint Model (GPU Accelerated)!");
            } catch (poseErr) {
                logMsg("yolov8n-pose.onnx failed on WebGPU/WebGL, attempting multi-thread WASM...", "warn");
                poseSession = await ort.InferenceSession.create('/static/yolov8n-pose.onnx', { executionProviders: ['wasm'] });
                logMsg("Successfully loaded YOLOv8-Pose (Multi-thread WASM)!");
            }

            logMsg("Loading 3D Lifter GCN (/static/models/3d_lifter_gcn.onnx)...");
            try {
                gcnSession = await ort.InferenceSession.create('/static/models/3d_lifter_gcn.onnx', { executionProviders: providers });
                logMsg("Successfully loaded 3D Lifter GCN ONNX Model (GPU Accelerated)!");
            } catch (gcnErr) {
                gcnSession = await ort.InferenceSession.create('/static/models/3d_lifter_gcn.onnx', { executionProviders: ['wasm'] });
                logMsg("Loaded 3D Lifter GCN ONNX Model (WASM)!");
            }

            statusText.innerText = "Running 3D Pose Inference (Active)";
        } catch (modelErr) {
            logMsg("Failed to load ONNX models: " + modelErr.message, "error");
            statusText.innerText = "Running 3D Inference Preview";
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
        let personFound = false;

        if (poseSession) {
            const inputTensor = preprocessVideoFrame();
            const feeds = {};
            feeds[poseSession.inputNames[0]] = inputTensor;
            const poseResults = await poseSession.run(feeds);
            const output = poseResults[poseSession.outputNames[0]].data;
            const numChannels = Math.floor(output.length / 8400);

            let maxArea = 0;
            let bestIdx = -1;

            for (let i = 0; i < 8400; i++) {
                const score = output[4 * 8400 + i];
                if (score > 0.20) {
                    const w = output[2 * 8400 + i];
                    const h = output[3 * 8400 + i];
                    const area = w * h;
                    if (area > maxArea) {
                        maxArea = area;
                        bestIdx = i;
                    }
                }
            }

            if (bestIdx >= 0 && numChannels === 56) {
                personFound = true;

                const getCOCO = (cIdx) => {
                    const kx = output[(5 + cIdx * 3 + 0) * 8400 + bestIdx];
                    const ky = output[(5 + cIdx * 3 + 1) * 8400 + bestIdx];
                    const kc = output[(5 + cIdx * 3 + 2) * 8400 + bestIdx];
                    return { x: kx, y: ky, conf: kc };
                };

                const nose = getCOCO(0);
                const lShoulder = getCOCO(5), rShoulder = getCOCO(6);
                const lElbow = getCOCO(7), rElbow = getCOCO(8);
                const lWrist = getCOCO(9), rWrist = getCOCO(10);
                const lHip = getCOCO(11), rHip = getCOCO(12);
                const lKnee = getCOCO(13), rKnee = getCOCO(14);
                const lAnkle = getCOCO(15), rAnkle = getCOCO(16);

                if (lHip && rHip && lShoulder && rShoulder) {
                    const pelvis2D = { x: (lHip.x + rHip.x) / 2.0, y: (lHip.y + rHip.y) / 2.0, conf: (lHip.conf + rHip.conf) / 2.0 };
                    const neck2D   = { x: (lShoulder.x + rShoulder.x) / 2.0, y: (lShoulder.y + rShoulder.y) / 2.0, conf: (lShoulder.conf + rShoulder.conf) / 2.0 };
                    const head2D   = nose && nose.conf > 0.2 ? nose : { x: neck2D.x, y: neck2D.y - 30.0, conf: 0.9 };

                    personCenterX = (pelvis2D.x) / MODEL_SIZE;
                    personCenterY = (pelvis2D.y) / MODEL_SIZE;

                    if (activeEngineMode === 'gcn_onnx' && gcnSession) {
                        // MODE B: Run 3D Lifter GCN ONNX Model (3d_lifter_gcn.onnx)
                        const sanitizeKP = (kp, parentKP, offsetX = 0, offsetY = 30) => {
                            // Only trigger fallback if keypoint is completely missing (conf <= 0.05 or x/y <= 0)
                            if (!kp || (kp.conf !== undefined && kp.conf <= 0.05) || kp.x <= 0 || kp.y <= 0) {
                                return { x: parentKP ? parentKP.x + offsetX : 320, y: parentKP ? parentKP.y + offsetY : 240, conf: 0.05 };
                            }
                            return kp;
                        };

                        const s_rHip = sanitizeKP(rHip, pelvis2D, -20, 20);
                        const s_rKnee = sanitizeKP(rKnee, s_rHip, 0, 50);
                        const s_rAnkle = sanitizeKP(rAnkle, s_rKnee, 0, 50);

                        const s_lHip = sanitizeKP(lHip, pelvis2D, 20, 20);
                        const s_lKnee = sanitizeKP(lKnee, s_lHip, 0, 50);
                        const s_lAnkle = sanitizeKP(lAnkle, s_lKnee, 0, 50);

                        const s_lShoulder = sanitizeKP(lShoulder, neck2D, 30, 0);
                        const s_lElbow = sanitizeKP(lElbow, s_lShoulder, 20, 30);
                        const s_lWrist = sanitizeKP(lWrist, s_lElbow, 20, 30);

                        const s_rShoulder = sanitizeKP(rShoulder, neck2D, -30, 0);
                        const s_rElbow = sanitizeKP(rElbow, s_rShoulder, -20, 30);
                        const s_rWrist = sanitizeKP(rWrist, s_rElbow, -20, 30);

                        const spine12D = { x: pelvis2D.x * 0.67 + neck2D.x * 0.33, y: pelvis2D.y * 0.67 + neck2D.y * 0.33, conf: 0.9 };
                        const spine22D = { x: pelvis2D.x * 0.33 + neck2D.x * 0.67, y: pelvis2D.y * 0.33 + neck2D.y * 0.67, conf: 0.9 };

                        const rawGCNNodes = [
                            pelvis2D,                                    // 0: Pelvis
                            s_rHip, s_rKnee, s_rAnkle,                   // 1, 2, 3: Right Leg
                            s_lHip, s_lKnee, s_lAnkle,                   // 4, 5, 6: Left Leg
                            spine12D, spine22D, neck2D, head2D,          // 7, 8, 9, 10: Spine & Head
                            s_lShoulder, s_lElbow, s_lWrist,             // 11, 12, 13: Left Arm
                            s_rShoulder, s_rElbow, s_rWrist              // 14, 15, 16: Right Arm
                        ];

                        const xs = rawGCNNodes.map(k => k.x);
                        const ys = rawGCNNodes.map(k => k.y);
                        const minX = Math.min(...xs);
                        const maxX = Math.max(...xs);
                        const minY = Math.min(...ys);
                        const maxY = Math.max(...ys);
                        const bw = Math.max(20.0, maxX - minX);
                        const bh = Math.max(20.0, maxY - minY);

                        const getNormKP = (kp) => ({
                            x: Math.max(0.0, Math.min(256.0, ((kp.x - minX) / bw) * 256.0)),
                            y: Math.max(0.0, Math.min(256.0, ((kp.y - minY) / bh) * 256.0)),
                            conf: kp.conf || 1.0
                        });

                        const gcnNodes = rawGCNNodes.map(kp => getNormKP(kp));

                        const nodesData = new Float32Array(17 * 5);
                        for (let i = 0; i < 17; i++) {
                            // Channel 0: X in [-1, 1], Channel 1: Y in [+1(Up), -1(Down)] (Matching dataset.py)
                            nodesData[i * 5 + 0] = (gcnNodes[i].x / 128.0) - 1.0;
                            nodesData[i * 5 + 1] = 1.0 - (gcnNodes[i].y / 128.0);
                            nodesData[i * 5 + 2] = 10.0;
                            nodesData[i * 5 + 3] = 10.0;
                            nodesData[i * 5 + 4] = gcnNodes[i].conf;
                        }

                        const tensorNodes = new ort.Tensor('float32', nodesData, [1, 17, 5]);
                        const tensorAdj = new ort.Tensor('float32', staticAdj, [1, 17, 17]);
                        const gcnResults = await gcnSession.run({ input_nodes: tensorNodes, input_adj: tensorAdj });
                        const outputData = gcnResults.output_joints.data;

                        // Model outputs Three.js native 3D coords [X, Y(height), Z(depth)]
                        const rootX = outputData[0 * 3 + 0];
                        const rootY = outputData[0 * 3 + 1];
                        const rootZ = outputData[0 * 3 + 2];

                        const GCN_DISPLAY_SCALE = 1.35; // Boosts ~1.36m span to ~1.83m real human height

                        for (let i = 0; i < 17; i++) {
                            // 1. Subtract Root (Pelvis) offset to anchor pelvis at origin (0, 0, 0)
                            const relX = (outputData[i * 3 + 0] - rootX) * GCN_DISPLAY_SCALE;
                            const relY = (outputData[i * 3 + 1] - rootY) * GCN_DISPLAY_SCALE;
                            const relZ = (outputData[i * 3 + 2] - rootZ) * GCN_DISPLAY_SCALE;

                            // 2. Metric outlier clamping [-2.5m, +2.5m]
                            const clampedX = Math.max(-2.5, Math.min(2.5, relX));
                            const clampedY = Math.max(-2.5, Math.min(2.5, relY));
                            const clampedZ = Math.max(-2.5, Math.min(2.5, relZ));

                            currentJoints3D[i].x = -clampedX;  // Mirror X for webcam
                            currentJoints3D[i].y = clampedY;   // Direct Three.js Y (Height up)
                            currentJoints3D[i].z = clampedZ;    // Direct Three.js Z (Depth)
                        }

                        if (performance.now() - lastLogTime > 2500) {
                            console.log("[ONNX GCN Raw Tensor Output (17x3)]", outputData);
                            console.log(`[ONNX GCN Root Joint 0 (Pelvis)] X:${currentJoints3D[0].x.toFixed(3)} Y:${currentJoints3D[0].y.toFixed(3)} Z:${currentJoints3D[0].z.toFixed(3)}`);
                        }

                    } else {
                        // MODE A: Direct 2D-to-3D Kinematic Pose Engine (Calibrated to ~1.8m vertical span)
                        const lift3D = (kp2D, parent2D, depthFactor = 0.0) => {
                            if (!kp2D) return { x: 0, y: 0, z: 0 };
                            const x3d = -((kp2D.x - pelvis2D.x) / 160.0);
                            const y3d = -((kp2D.y - pelvis2D.y) / 160.0);
                            const torsoLen = Math.hypot(neck2D.x - pelvis2D.x, neck2D.y - pelvis2D.y) || 100.0;
                            const z3d = (depthFactor * (torsoLen / 100.0));
                            return { x: x3d, y: y3d, z: z3d };
                        };

                        currentJoints3D[0]  = { x: 0, y: 0, z: 0 };
                        currentJoints3D[1]  = lift3D(rHip, pelvis2D, 0.05);
                        currentJoints3D[2]  = lift3D(rKnee, rHip, 0.1);
                        currentJoints3D[3]  = lift3D(rAnkle, rKnee, 0.0);
                        currentJoints3D[4]  = lift3D(lHip, pelvis2D, -0.05);
                        currentJoints3D[5]  = lift3D(lKnee, lHip, 0.1);
                        currentJoints3D[6]  = lift3D(lAnkle, lKnee, 0.0);

                        const spine12D = { x: pelvis2D.x * 0.67 + neck2D.x * 0.33, y: pelvis2D.y * 0.67 + neck2D.y * 0.33 };
                        const spine22D = { x: pelvis2D.x * 0.33 + neck2D.x * 0.67, y: pelvis2D.y * 0.33 + neck2D.y * 0.67 };

                        currentJoints3D[7]  = lift3D(spine12D, pelvis2D, 0.02);
                        currentJoints3D[8]  = lift3D(spine22D, pelvis2D, 0.04);
                        currentJoints3D[9]  = lift3D(neck2D, pelvis2D, 0.05);
                        currentJoints3D[10] = lift3D(head2D, neck2D, 0.08);

                        currentJoints3D[11] = lift3D(lShoulder, neck2D, -0.05);
                        currentJoints3D[12] = lift3D(lElbow, lShoulder, 0.15);
                        currentJoints3D[13] = lift3D(lWrist, lElbow, 0.2);

                        currentJoints3D[14] = lift3D(rShoulder, neck2D, 0.05);
                        currentJoints3D[15] = lift3D(rElbow, rShoulder, 0.15);
                        currentJoints3D[16] = lift3D(rWrist, rElbow, 0.2);
                    }
                }
            }
        }

        const elapsed = (performance.now() - tStart).toFixed(1);
        if (performance.now() - lastLogTime > 2500) {
            const modeName = activeEngineMode === 'gcn_onnx' ? '3D Lifter GCN (ONNX)' : '3D Kinematic Engine';
            const statusStr = personFound ? `Target Tracked (${modeName})` : `Searching`;
            const dx = (Math.max(...currentJoints3D.map(j => j.x)) - Math.min(...currentJoints3D.map(j => j.x))).toFixed(2);
            const dy = (Math.max(...currentJoints3D.map(j => j.y)) - Math.min(...currentJoints3D.map(j => j.y))).toFixed(2);
            const dz = (Math.max(...currentJoints3D.map(j => j.z)) - Math.min(...currentJoints3D.map(j => j.z))).toFixed(2);

            logMsg(`[3D Engine] ${statusStr} | Latency: ${elapsed}ms | Position: (${personCenterX.toFixed(2)}, ${personCenterY.toFixed(2)}) | 3D Span -> X:[${dx}m] Y:[${dy}m] Z:[${dz}m]`);
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
        
        frames++;
        if (time - lastTime >= 1000) {
            fpsText.innerText = frames;
            frames = 0;
            lastTime = time;
        }

        if (poseSession && !isInferringPipeline) {
            runEndToEndPipeline(time);
        }

        const worldX = -(personCenterX - 0.5) * 4.5;
        const worldY = -(personCenterY - 0.5) * 3.5;

        for (let i = 0; i < jointCount; i++) {
            const px = currentJoints3D[i].x * 1.5 + worldX;
            const py = currentJoints3D[i].y * 1.5 + worldY;
            const pz = currentJoints3D[i].z * 1.5;

            dummy.position.set(px, py, pz);
            dummy.updateMatrix();
            instancedMesh.setMatrixAt(i, dummy.matrix);
        }
        instancedMesh.instanceMatrix.needsUpdate = true;

        for (let b = 0; b < bonePairs.length; b++) {
            const [i, j] = bonePairs[b];

            const p1x = currentJoints3D[i].x * 1.5 + worldX;
            const p1y = currentJoints3D[i].y * 1.5 + worldY;
            const p1z = currentJoints3D[i].z * 1.5;

            const p2x = currentJoints3D[j].x * 1.5 + worldX;
            const p2y = currentJoints3D[j].y * 1.5 + worldY;
            const p2z = currentJoints3D[j].z * 1.5;

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

window.addEventListener('resize', () => {
    camera.aspect = 640 / 480;
    camera.updateProjectionMatrix();
    renderer.setSize(640, 480);
});

initEngine();
