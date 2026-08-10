import streamlit as st
import sys
import socket
import mimetypes

# Fix for Streamlit/Tornado static file serving defaulting to text/plain on macOS/Windows
mimetypes.add_type('text/html', '.html')
mimetypes.add_type('application/javascript', '.js')

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

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

HOST_IP = get_local_ip()

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

# Performance Metrics Sidebar Card
try:
    from utils.metrics_logger import load_metrics
    metrics_data = load_metrics()
    with st.sidebar.expander("📊 Performance Metrics Report", expanded=False):
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Precision", metrics_data.get("detection_precision", "N/A"))
        col_m2.metric("Recall", metrics_data.get("detection_recall", "N/A"))
        st.metric("F1 Score", metrics_data.get("f1_score", "N/A"))
        st.metric("Avg FPS", f"{metrics_data.get('average_fps_per_stream', 'N/A')} FPS")
        st.metric("Avg Latency", f"{metrics_data.get('average_latency_ms', 'N/A')} ms")
        if "last_updated" in metrics_data:
            st.caption(f"Updated: {metrics_data['last_updated']}")
except Exception as e:
    logging.error(f"Error loading performance metrics: {e}")

mode = st.sidebar.radio("Select View Mode", ["2D Multi-Stream (Audit Mode)", "2D Webcam Edge AI (ONNX/WASM)", "3D Advanced Engine (WASM/WebGL)"])

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
        import psutil
        
        # Kill any orphaned python workers from previous runs to prevent CPU starvation and stream conflicts
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and 'run_inference_worker.py' in proc.info['cmdline']:
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Stream 1 Worker
        if stream1_file:
            t1 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t1.write(stream1_file.read())
            t1.close()
            
            # Spawn totally isolated python process using exact python executable
            subprocess.Popen([sys.executable, 'run_inference_worker.py', '--stream_id', '1', '--video_path', t1.name, '--conf', str(conf_threshold)])
            
            with col1:
                st.markdown("### Stream 1 Live Gateway Feed")
                iframe_html = f"""
                <html><body style="margin:0;padding:0;background:transparent;">
                    <div style="position:relative; display:inline-block; width:100%;">
                        <video id="vid" src="http://{HOST_IP}:8080/video?path={t1.name}" muted playsinline preload="auto" style="display:block; width:100%; height:auto; border-radius:8px;"></video>
                        <canvas id="overlay" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"></canvas>
                        <div id="proc" style="position:absolute; top:0; left:0; width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.8); color:white; font-family:sans-serif; flex-direction:column; gap:8px; border-radius:8px;">
                            <div style="font-size:28px;">⏳</div>
                            <div>Processing with YOLO AI...</div>
                            <div id="fc" style="font-size:13px;opacity:0.6;">0 frames</div>
                        </div>
                    </div>
                    <script>
                        const vid = document.getElementById('vid');
                        const canvas = document.getElementById('overlay');
                        const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
                        const proc = document.getElementById('proc');
                        const fc = document.getElementById('fc');
                        vid.addEventListener('loadedmetadata', () => {{ if (vid.videoWidth > 0) {{ canvas.width = vid.videoWidth; canvas.height = vid.videoHeight; }} }});
                        const ws = new WebSocket('ws://{HOST_IP}:8080/view_stream?stream=1');
                        let frameBuffer = [];
                        ws.onmessage = function(event) {{
                            const handleData = function(data) {{
                                if (data.status === 'done') {{
                                    proc.style.display = 'none';
                                    vid.currentTime = 0;
                                    vid.play().catch(function() {{
                                        proc.innerHTML = '<div style="cursor:pointer;font-size:48px;text-align:center;">▶️<br><span style="font-size:14px">Tap to Play</span></div>';
                                        proc.style.display = 'flex';
                                        proc.onclick = function() {{ vid.play(); proc.style.display = 'none'; }};
                                    }});
                                    return;
                                }}
                                frameBuffer.push(data);
                                fc.innerText = frameBuffer.length + ' frames';
                                if (frameBuffer.length > 10000) frameBuffer.shift();
                            }};
                            if (typeof event.data === 'string') {{ handleData(JSON.parse(event.data)); }}
                            else {{ var r = new FileReader(); r.onload = function() {{ handleData(JSON.parse(r.result)); }}; r.readAsText(event.data); }}
                        }};
                        function draw() {{
                            if (vid.videoWidth > 0 && (canvas.width !== vid.videoWidth || canvas.height !== vid.videoHeight)) {{ canvas.width = vid.videoWidth; canvas.height = vid.videoHeight; }}
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            if (frameBuffer.length > 0 && !vid.paused) {{
                                var ct = vid.currentTime, st2 = ct;
                                var maxT = frameBuffer[frameBuffer.length - 1].timestamp;
                                if (st2 > maxT) st2 = maxT;
                                var f1 = null, f2 = null, low = 0, high = frameBuffer.length - 1;
                                while (low <= high) {{ var mid = (low + high) >> 1; if (frameBuffer[mid].timestamp <= st2) {{ f1 = frameBuffer[mid]; low = mid + 1; }} else {{ high = mid - 1; }} }}
                                var f1Idx = low - 1;
                                if (f1 && f1Idx + 1 < frameBuffer.length) f2 = frameBuffer[f1Idx + 1];
                                if (f1) {{
                                    f1.boxes.forEach(function(b1) {{
                                        var x1=b1.bbox[0],y1=b1.bbox[1],x2=b1.bbox[2],y2=b1.bbox[3];
                                        if (f2) {{ var b2 = f2.boxes.find(function(b){{ return b.id===b1.id; }}); if (b2) {{ var p=Math.max(0,Math.min(1,(st2-f1.timestamp)/(f2.timestamp-f1.timestamp))); x1+=(b2.bbox[0]-x1)*p; y1+=(b2.bbox[1]-y1)*p; x2+=(b2.bbox[2]-x2)*p; y2+=(b2.bbox[3]-y2)*p; }} }}
                                        ctx.strokeStyle='#00FF00'; ctx.lineWidth=4; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
                                        ctx.fillStyle='#00FF00'; ctx.font='20px Arial'; ctx.fillText('ID:'+b1.id,x1,y1-8);
                                    }});
                                    ctx.fillStyle='#FF0000'; ctx.font='26px Arial'; ctx.fillText('Total: '+f1.total,15,35);
                                }}
                            }}
                            requestAnimationFrame(draw);
                        }}
                        requestAnimationFrame(draw);
                    </script>
                </body></html>
                """
                st.components.v1.html(iframe_html, height=420)

        # Stream 2 Worker
        if stream2_file:
            t2 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t2.write(stream2_file.read())
            t2.close()
            
            # Spawn totally isolated python process using exact python executable
            subprocess.Popen([sys.executable, 'run_inference_worker.py', '--stream_id', '2', '--video_path', t2.name, '--conf', str(conf_threshold)])
            
            with col2:
                st.markdown("### Stream 2 Live Gateway Feed")
                iframe_html = f"""
                <html><body style="margin:0;padding:0;background:transparent;">
                    <div style="position:relative; display:inline-block; width:100%;">
                        <video id="vid" src="http://{HOST_IP}:8080/video?path={t2.name}" muted playsinline preload="auto" style="display:block; width:100%; height:auto; border-radius:8px;"></video>
                        <canvas id="overlay" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"></canvas>
                        <div id="proc" style="position:absolute; top:0; left:0; width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.8); color:white; font-family:sans-serif; flex-direction:column; gap:8px; border-radius:8px;">
                            <div style="font-size:28px;">⏳</div>
                            <div>Processing with YOLO AI...</div>
                            <div id="fc" style="font-size:13px;opacity:0.6;">0 frames</div>
                        </div>
                    </div>
                    <script>
                        const vid = document.getElementById('vid');
                        const canvas = document.getElementById('overlay');
                        const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
                        const proc = document.getElementById('proc');
                        const fc = document.getElementById('fc');
                        vid.addEventListener('loadedmetadata', () => {{ if (vid.videoWidth > 0) {{ canvas.width = vid.videoWidth; canvas.height = vid.videoHeight; }} }});
                        const ws = new WebSocket('ws://{HOST_IP}:8080/view_stream?stream=2');
                        let frameBuffer = [];
                        ws.onmessage = function(event) {{
                            const handleData = function(data) {{
                                if (data.status === 'done') {{
                                    proc.style.display = 'none';
                                    vid.currentTime = 0;
                                    vid.play().catch(function() {{
                                        proc.innerHTML = '<div style="cursor:pointer;font-size:48px;text-align:center;">▶️<br><span style="font-size:14px">Tap to Play</span></div>';
                                        proc.style.display = 'flex';
                                        proc.onclick = function() {{ vid.play(); proc.style.display = 'none'; }};
                                    }});
                                    return;
                                }}
                                frameBuffer.push(data);
                                fc.innerText = frameBuffer.length + ' frames';
                                if (frameBuffer.length > 10000) frameBuffer.shift();
                            }};
                            if (typeof event.data === 'string') {{ handleData(JSON.parse(event.data)); }}
                            else {{ var r = new FileReader(); r.onload = function() {{ handleData(JSON.parse(r.result)); }}; r.readAsText(event.data); }}
                        }};
                        function draw() {{
                            if (vid.videoWidth > 0 && (canvas.width !== vid.videoWidth || canvas.height !== vid.videoHeight)) {{ canvas.width = vid.videoWidth; canvas.height = vid.videoHeight; }}
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            if (frameBuffer.length > 0 && !vid.paused) {{
                                var ct = vid.currentTime, st2 = ct;
                                var maxT = frameBuffer[frameBuffer.length - 1].timestamp;
                                if (st2 > maxT) st2 = maxT;
                                var f1 = null, f2 = null, low = 0, high = frameBuffer.length - 1;
                                while (low <= high) {{ var mid = (low + high) >> 1; if (frameBuffer[mid].timestamp <= st2) {{ f1 = frameBuffer[mid]; low = mid + 1; }} else {{ high = mid - 1; }} }}
                                var f1Idx = low - 1;
                                if (f1 && f1Idx + 1 < frameBuffer.length) f2 = frameBuffer[f1Idx + 1];
                                if (f1) {{
                                    f1.boxes.forEach(function(b1) {{
                                        var x1=b1.bbox[0],y1=b1.bbox[1],x2=b1.bbox[2],y2=b1.bbox[3];
                                        if (f2) {{ var b2 = f2.boxes.find(function(b){{ return b.id===b1.id; }}); if (b2) {{ var p=Math.max(0,Math.min(1,(st2-f1.timestamp)/(f2.timestamp-f1.timestamp))); x1+=(b2.bbox[0]-x1)*p; y1+=(b2.bbox[1]-y1)*p; x2+=(b2.bbox[2]-x2)*p; y2+=(b2.bbox[3]-y2)*p; }} }}
                                        ctx.strokeStyle='#00FF00'; ctx.lineWidth=4; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
                                        ctx.fillStyle='#00FF00'; ctx.font='20px Arial'; ctx.fillText('ID:'+b1.id,x1,y1-8);
                                    }});
                                    ctx.fillStyle='#FF0000'; ctx.font='26px Arial'; ctx.fillText('Total: '+f1.total,15,35);
                                }}
                            }}
                            requestAnimationFrame(draw);
                        }}
                        requestAnimationFrame(draw);
                    </script>
                </body></html>
                """
                st.components.v1.html(iframe_html, height=420)
                
        st.success("🚀 Microservices successfully spawned in background processes! Streamlit is free. Videos are streaming via the Go Gateway.")

elif mode == "2D Webcam Edge AI (ONNX/WASM)":
    st.header("2D Webcam Edge AI (ONNX Runtime Web)")
    st.markdown("This component runs YOLOv8n purely inside the browser using **ONNX Runtime Web** and WebAssembly. Zero network latency.")
    
    try:
        with open("frontend_2d/index.html", "r") as f:
            html_code = f.read()
            
        with open("frontend_2d/app.js", "r") as f_js:
            js_code = f_js.read()

        # Inline the JS into the HTML
        html_code = html_code.replace('<script src="app.js"></script>', f'<script>\n{js_code}\n</script>')
        
        # Write to static/ folder and serve via a real src= iframe (NOT srcdoc).
        # This gives the iframe origin http://localhost:8501 instead of null,
        # so fetch('/app/static/yolov8n.onnx') is same-origin in both Safari and Chrome.
        import os
        os.makedirs("static", exist_ok=True)
        with open("static/webcam.html", "w") as f:
            f.write(html_code)
        
        iframe_host = "localhost"
        st.markdown(
            f'<iframe src="http://{iframe_host}:8080/static/webcam.html" width="100%" height="550" allow="camera; microphone; display-capture" style="border:none; border-radius:8px;"></iframe>',
            unsafe_allow_html=True
        )
        st.info("💡 **Security Tip:** Modern browsers require `localhost` or `https://` to grant webcam access.")
    except FileNotFoundError:
        st.warning("Frontend 2D ONNX component not built yet.")

else:
    st.header("Advanced 3D WebGL Engine")
    st.markdown("This component runs the 3D Lifter GCN, WASM mathematical abstraction, and Three.js rendering completely within the browser.")
    # Auto-check and download YOLOv11-nano ONNX if missing
    try:
        from utils.ensure_yolo11n import ensure_yolo11n_onnx
        ensure_yolo11n_onnx()
    except Exception as e:
        logging.error(f"Error ensuring yolo11n.onnx: {e}")

    try:
        with open("frontend/index.html", "r") as f:
            html_code = f.read()
            
        with open("frontend/app.js", "r") as f_js:
            js_code = f_js.read()
            
        html_code = html_code.replace('<script src="app.js"></script>', f'<script>\n{js_code}\n</script>')
        
        import os
        os.makedirs("static", exist_ok=True)
        with open("static/engine3d.html", "w") as f:
            f.write(html_code)
        
        iframe_host = "localhost"
        st.markdown(
            f'<iframe src="http://{iframe_host}:8080/static/engine3d.html" width="100%" height="550" allow="camera; microphone; display-capture" style="border:none; border-radius:8px;"></iframe>',
            unsafe_allow_html=True
        )
        st.info("💡 **Camera Access Tip:** Modern browsers block webcam access (`navigator.mediaDevices`) on insecure connections. To use this 3D feature, you must access the app via `http://localhost:8501` or a secure `https://` connection.")
    except FileNotFoundError:
        st.warning("Frontend component not built yet.")
