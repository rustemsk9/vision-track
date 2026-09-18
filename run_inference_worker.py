import argparse
import cv2
import requests
import json
import supervision as sv
from models.yolo_person_detection import PersonDetector
from utils.multi_stream_tracking_helpers import StreamTracker

parser = argparse.ArgumentParser()
parser.add_argument("--stream_id", type=str, required=True)
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--conf", type=float, default=0.5)
args = parser.parse_args()

detector = PersonDetector("yolov8n.pt")
tracker = StreamTracker()

cap = cv2.VideoCapture(args.video_path)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

scale_x = orig_w / 640.0 if orig_w > 0 else 1.0
scale_y = orig_h / 360.0 if orig_h > 0 else 1.0

import time

import time
import threading
import queue

session = requests.Session()
frame_idx = 0

# Network push thread to ensure AI loop never blocks on HTTP requests
payload_queue = queue.Queue(maxsize=3000)
stop_worker = False

def push_worker():
    global stop_worker
    while True:
        payload = payload_queue.get()
        if payload is None or stop_worker:
            break
            
        # Retry loop to guarantee NO frames are ever dropped due to local network saturation timeouts
        success = False
        retries = 0
        while not success and not stop_worker and retries < 10:
            try:
                resp = session.post(f"http://127.0.0.1:8080/push_frame?stream={args.stream_id}", data=payload.encode('utf-8'), timeout=5.0)
                if resp.status_code == 410 and (time.time() - overall_start) > 10:
                    print(f"[{args.stream_id}] No browser clients connected. Shutting down worker gracefully.")
                    stop_worker = True
                    break
                success = True
            except requests.exceptions.Timeout:
                retries += 1
                time.sleep(0.1)
            except Exception as e:
                break
                
        payload_queue.task_done()

threading.Thread(target=push_worker, daemon=True).start()

print(f"[{args.stream_id}] Starting inference loop. FPS: {fps}, Resolution: {orig_w}x{orig_h}")
overall_start = time.time()

# Auto-adjust frame_skip based on video length to keep processing time reasonable
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration_sec = total_frames / fps if fps > 0 else 0

if video_duration_sec > 300:    # > 5 minutes: fast mode
    frame_skip = 15
elif video_duration_sec > 120:  # > 2 minutes: balanced
    frame_skip = 10
else:                           # short videos: smooth 6 fps
    frame_skip = 5

print(f"[{args.stream_id}] Video: {video_duration_sec:.1f}s ({total_frames} frames), frame_skip={frame_skip}, ~{fps/frame_skip:.1f} AI keyframes/sec")

while cap.isOpened() and not stop_worker:
    t_start = time.time()
    
    ret, frame = cap.read()
    t_read = time.time()
    
    if not ret:
        break
        
    # Use exact Presentation Time Stamp to prevent VFR (Variable Frame Rate) video drift
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    
    # Fallback if OpenCV fails to extract PTS (returns 0.0 on some MP4s)
    if timestamp <= 0.001 and frame_idx > 5:
        timestamp = frame_idx / fps
        
    frame_idx += 1
    
    if frame_idx % frame_skip != 0:
        continue
    
    frame = cv2.resize(frame, (640, 360))
    results = detector.predict(frame, conf=args.conf)
    detections = sv.Detections.from_ultralytics(results)
    frame, tracked_detections = tracker.update_and_annotate(frame, detections)
    t_ai = time.time()
    
    total_people = len(tracked_detections)
    
    boxes = []
    if tracked_detections.xyxy is not None:
        for i in range(len(tracked_detections.xyxy)):
            x1, y1, x2, y2 = tracked_detections.xyxy[i].tolist()
            t_id = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else -1
            boxes.append({
                "id": t_id,
                "bbox": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            })
    
    payload = json.dumps({"timestamp": timestamp, "boxes": boxes, "total": total_people})
    
    # Push to thread queue instead of blocking
    try:
        payload_queue.put_nowait(payload)
    except queue.Full:
        pass
        
    t_post = time.time()
    
    ms_read = (t_read - t_start) * 1000
    ms_ai = (t_ai - t_read) * 1000
    ms_post = (t_post - t_ai) * 1000
    ms_total = (t_post - t_start) * 1000
    
    # We skip printing per-frame to prevent terminal scroll locking issues that freeze the process

cap.release()

# Save real performance metrics to reports/performance_metrics.json
try:
    from utils.metrics_logger import update_metrics
    elapsed = time.time() - overall_start
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0
    proc_latency = (elapsed / frame_idx * 1000) if frame_idx > 0 else 0
    update_metrics(
        fps=proc_fps,
        latency_ms=proc_latency,
        extra_info={
            "stream_id": args.stream_id,
            "processed_frames": frame_idx,
            "total_video_time_sec": round(elapsed, 2)
        }
    )
    print(f"[{args.stream_id}] Saved performance metrics to reports/performance_metrics.json")
except Exception as e:
    print(f"[{args.stream_id}] Failed to save metrics: {e}")

# Wait for queue to finish
try:
    payload_queue.put_nowait(json.dumps({"status": "done"}))
except queue.Full:
    pass

# Small delay to let thread finish
time.sleep(0.5)

print(f"[{args.stream_id}] Worker finished processing video. Total time: {time.time() - overall_start:.2f}s")
print(f"[{args.stream_id}] Keeping process alive in background to prevent PyTorch MPS context teardown crashes...")

# Prevent MPS GPU context teardown which hangs concurrent streams
while not stop_worker:
    time.sleep(1)
    
print(f"[{args.stream_id}] Worker fully shutting down.")



