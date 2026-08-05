import streamlit as st
import streamlit.components.v1 as components
import torch
import cv2
import json
import logging
from pathlib import Path
import supervision as sv
import tempfile
from models.yolo_person_detection import PersonDetector
from utils.multi_stream_tracking_helpers import StreamTracker
from utils.counting_logic import ROICounter

# Ensure log directory exists
Path("logs").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
logging.basicConfig(filename='logs/app_errors.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

st.set_page_config(layout="wide", page_title="VisionTrack - Hybrid 3D Pose", page_icon="🎥")

st.title("VisionTrack: 3D Human Pose Estimation & Multi-Stream Tracking")

# Hardware check
try:
    if torch.cuda.is_available():
        hardware_status = "🟢 GPU (CUDA)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        hardware_status = "🔵 GPU (Apple Metal/MPS)"
    else:
        hardware_status = "🟡 CPU (Fallback Mode)"
except Exception as e:
    logging.error(f"Error checking Hardware: {e}")
    hardware_status = "🟡 CPU (Fallback Mode)"

st.sidebar.markdown(f"**Hardware:** {hardware_status}")

mode = st.sidebar.radio("Select View Mode", ["2D Multi-Stream (Audit Mode)", "3D Advanced Engine (WASM/WebGL)"])

if mode == "2D Multi-Stream (Audit Mode)":
    st.header("2D Multi-Stream Tracking")
    # Toggles for features
    import time
    st.sidebar.subheader("Stream Settings")
    enable_detection = st.sidebar.checkbox("Enable Detection", value=True)
    enable_tracking = st.sidebar.checkbox("Enable Tracking", value=True)
    enable_counting = st.sidebar.checkbox("Enable Total People Count", value=True)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    st.sidebar.subheader("Video Sources")
    stream1_file = st.sidebar.file_uploader("Upload Stream 1", type=['mp4', 'avi', 'mov'])
    stream2_file = st.sidebar.file_uploader("Upload Stream 2", type=['mp4', 'avi', 'mov'])
    
    @st.cache_resource
    def load_detector():
        return PersonDetector("yolov8n.pt")
        
    detector = load_detector()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stream 1")
        if stream1_file:
            st.video(stream1_file)
        else:
            st.info("Upload a video file in the sidebar to start Stream 1.")
            
    with col2:
        st.subheader("Stream 2")
        if stream2_file:
            st.video(stream2_file)
        else:
            st.info("Upload a video file in the sidebar to start Stream 2.")
            
    if (stream1_file or stream2_file) and st.button("Start Microservice Architecture"):
        import subprocess
        import html
        
        # Stream 1 Worker
        if stream1_file:
            t1 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t1.write(stream1_file.read())
            t1.close()
            
            # Spawn totally isolated python process
            subprocess.Popen(['python', 'run_inference_worker.py', '--stream_id', '1', '--video_path', t1.name, '--conf', str(conf_threshold)])
            
            with col1:
                st.markdown("### Stream 1 Live Gateway Feed")
                iframe_html = f"""
                <html><body style="margin:0;padding:0;background:#000;position:relative;">
                    <video id="vid" src="http://localhost:8080/video?path={t1.name}" controls muted style="position:absolute; width:100%; top:0; left:0; border-radius: 8px;"></video>
                    <canvas id="overlay" style="position:absolute; width:100%; top:0; left:0; pointer-events:none;"></canvas>
                    <script>
                        // Safely resolve the host from within the srcdoc sandbox
                        let host = "localhost";
                        try {{ host = window.parent.location.hostname || window.location.hostname; }} catch(e) {{}}
                        if (!host) host = "localhost";
                        
                        console.log("Stream 1: Attempting to load video from Go Gateway...");
                        document.getElementById('vid').src = 'http://' + host + ':8080/video?path={t1.name}';
                        
                        const vid = document.getElementById('vid');
                        const canvas = document.getElementById('overlay');
                        const ctx = canvas.getContext('2d');
                        
                        // Sync canvas resolution to original video resolution
                        vid.addEventListener('loadedmetadata', () => {{
                            console.log("Stream 1: Video metadata loaded successfully! Resolution:", vid.videoWidth, "x", vid.videoHeight);
                            canvas.width = vid.videoWidth;
                            canvas.height = vid.videoHeight;
                        }});
                        
                        vid.addEventListener('error', (e) => console.error("Stream 1: Video load error!", e));
                        
                        console.log("Stream 1: Connecting to WebSocket for AI coordinates...");
                        const ws = new WebSocket('ws://' + host + ':8080/view_stream?stream=1');
                        let frameBuffer = [];
                        
                        ws.onmessage = function(event) {{
                            // Receive JSON payload instead of JPEGs
                            if (typeof event.data === "string") {{
                                const data = JSON.parse(event.data);
                                frameBuffer.push(data);
                                if (frameBuffer.length > 500) frameBuffer.shift(); 
                            }} else {{
                                const reader = new FileReader();
                                reader.onload = function() {{
                                    const data = JSON.parse(reader.result);
                                    frameBuffer.push(data);
                                    if (frameBuffer.length > 500) frameBuffer.shift();
                                }};
                                reader.readAsText(event.data);
                            }}
                            
                            // SYNC LOCK: Start video only after AI data begins arriving
                            if (vid.paused && frameBuffer.length > 5) {{
                                console.log("Stream 1: AI Data buffered. Starting synced playback!");
                                vid.play();
                            }}
                        }};
                        
                        function draw() {{
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            const currentTime = vid.currentTime;
                            
                            // Find closest AI frame by timestamp
                            let closestFrame = null;
                            let minDiff = 999;
                            for (let i = 0; i < frameBuffer.length; i++) {{
                                const diff = Math.abs(frameBuffer[i].timestamp - currentTime);
                                if (diff < minDiff) {{
                                    minDiff = diff;
                                    closestFrame = frameBuffer[i];
                                }}
                            }}
                            
                            if (closestFrame && minDiff < 0.3) {{ 
                                closestFrame.boxes.forEach(b => {{
                                    const [x1, y1, x2, y2] = b.bbox;
                                    ctx.strokeStyle = '#00FF00';
                                    ctx.lineWidth = 4;
                                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                                    
                                    ctx.fillStyle = '#00FF00';
                                    ctx.font = '24px Arial';
                                    ctx.fillText('ID: ' + b.id, x1, y1 - 10);
                                }});
                                ctx.fillStyle = '#FF0000';
                                ctx.font = '30px Arial';
                                ctx.fillText('Total: ' + closestFrame.total, 20, 40);
                            }}
                            requestAnimationFrame(draw);
                        }}
                        requestAnimationFrame(draw);
                    </script>
                </body></html>
                """
                components.html(iframe_html, height=450)

        # Stream 2 Worker
        if stream2_file:
            t2 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t2.write(stream2_file.read())
            t2.close()
            
            # Spawn totally isolated python process
            subprocess.Popen(['python', 'run_inference_worker.py', '--stream_id', '2', '--video_path', t2.name, '--conf', str(conf_threshold)])
            
            with col2:
                st.markdown("### Stream 2 Live Gateway Feed")
                iframe_html = f"""
                <html><body style="margin:0;padding:0;background:#000;position:relative;">
                    <video id="vid" src="http://localhost:8080/video?path={t2.name}" controls muted style="position:absolute; width:100%; top:0; left:0; border-radius: 8px;"></video>
                    <canvas id="overlay" style="position:absolute; width:100%; top:0; left:0; pointer-events:none;"></canvas>
                    <script>
                        // Safely resolve the host from within the srcdoc sandbox
                        let host = "localhost";
                        try {{ host = window.parent.location.hostname || window.location.hostname; }} catch(e) {{}}
                        if (!host) host = "localhost";
                        
                        console.log("Stream 2: Attempting to load video from Go Gateway...");
                        document.getElementById('vid').src = 'http://' + host + ':8080/video?path={t2.name}';
                        
                        const vid = document.getElementById('vid');
                        const canvas = document.getElementById('overlay');
                        const ctx = canvas.getContext('2d');
                        
                        // Sync canvas resolution to original video resolution
                        vid.addEventListener('loadedmetadata', () => {{
                            console.log("Stream 2: Video metadata loaded successfully! Resolution:", vid.videoWidth, "x", vid.videoHeight);
                            canvas.width = vid.videoWidth;
                            canvas.height = vid.videoHeight;
                        }});
                        
                        vid.addEventListener('error', (e) => console.error("Stream 2: Video load error!", e));
                        
                        console.log("Stream 2: Connecting to WebSocket for AI coordinates...");
                        const ws = new WebSocket('ws://' + host + ':8080/view_stream?stream=2');
                        let frameBuffer = [];
                        
                        ws.onmessage = function(event) {{
                            // Receive JSON payload instead of JPEGs
                            if (typeof event.data === "string") {{
                                const data = JSON.parse(event.data);
                                frameBuffer.push(data);
                                if (frameBuffer.length > 500) frameBuffer.shift(); 
                            }} else {{
                                const reader = new FileReader();
                                reader.onload = function() {{
                                    const data = JSON.parse(reader.result);
                                    frameBuffer.push(data);
                                    if (frameBuffer.length > 500) frameBuffer.shift();
                                }};
                                reader.readAsText(event.data);
                            }}
                            
                            // SYNC LOCK: Start video only after AI data begins arriving
                            if (vid.paused && frameBuffer.length > 5) {{
                                console.log("Stream 2: AI Data buffered. Starting synced playback!");
                                vid.play();
                            }}
                        }};
                        
                        function draw() {{
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            const currentTime = vid.currentTime;
                            
                            // Find closest AI frame by timestamp
                            let closestFrame = null;
                            let minDiff = 999;
                            for (let i = 0; i < frameBuffer.length; i++) {{
                                const diff = Math.abs(frameBuffer[i].timestamp - currentTime);
                                if (diff < minDiff) {{
                                    minDiff = diff;
                                    closestFrame = frameBuffer[i];
                                }}
                            }}
                            
                            if (closestFrame && minDiff < 0.3) {{ 
                                closestFrame.boxes.forEach(b => {{
                                    const [x1, y1, x2, y2] = b.bbox;
                                    ctx.strokeStyle = '#00FF00';
                                    ctx.lineWidth = 4;
                                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                                    
                                    ctx.fillStyle = '#00FF00';
                                    ctx.font = '24px Arial';
                                    ctx.fillText('ID: ' + b.id, x1, y1 - 10);
                                }});
                                ctx.fillStyle = '#FF0000';
                                ctx.font = '30px Arial';
                                ctx.fillText('Total: ' + closestFrame.total, 20, 40);
                            }}
                            requestAnimationFrame(draw);
                        }}
                        requestAnimationFrame(draw);
                    </script>
                </body></html>
                """
                components.html(iframe_html, height=450)
                
        st.success("🚀 Microservices successfully spawned in background processes! Streamlit is free. Videos are streaming via the Go Gateway.")

else:
    import html
    st.header("Advanced 3D WebGL Engine")
    st.markdown("This component runs the 3D Lifter GCN, WASM mathematical abstraction, and Three.js rendering completely within the browser.")
    # Embedded HTML frontend for WebGL/Three.js
    try:
        with open("frontend/index.html", "r") as f:
            html_code = f.read()
            
        with open("frontend/app.js", "r") as f_js:
            js_code = f_js.read()
            
        # Strip comments and newlines to prevent Streamlit's Markdown parser from 
        # breaking the iframe tags and leaking raw Javascript onto the page
        import re
        js_code = re.sub(r'//.*', '', js_code)
        js_code = js_code.replace('\n', ' ')
            
        # Inline the JS script so it can run inside the iframe without external file requests
        html_code = html_code.replace('<script src="app.js"></script>', f'<script>{js_code}</script>')
        html_code = html_code.replace('\n', ' ')
        
        # Use a srcdoc iframe to inherit the parent origin and allow camera access
        srcdoc = html.escape(html_code)
        iframe_html = f'<iframe srcdoc="{srcdoc}" width="100%" height="650px" allow="camera; microphone" style="border:none; border-radius: 8px;"></iframe>'
        
        st.markdown(iframe_html, unsafe_allow_html=True)
        
        st.info("💡 **Camera Access Tip:** Modern browsers block webcam access (`navigator.mediaDevices`) on insecure connections. To use this 3D feature, you must access the app via `http://localhost:8501` or a secure `https://` connection.")
    except FileNotFoundError:
        st.warning("Frontend component not built yet. Proceeding to Phase 3 & 4 implementation.")

