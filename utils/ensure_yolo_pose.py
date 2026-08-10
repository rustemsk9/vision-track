import os
import shutil

def ensure_yolo_pose_onnx():
    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static/yolov8n-pose.onnx"))
    
    if os.path.exists(target_path):
        print(f"yolov8n-pose.onnx already exists at: {target_path}")
        return target_path

    print("yolov8n-pose.onnx not found in static/. Downloading and exporting YOLOv8-Pose...")
    try:
        from ultralytics import YOLO
        
        # Download and load YOLOv8-pose
        model = YOLO("yolov8n-pose.pt")
        
        # Export to ONNX with 640x640 input resolution
        exported_file = model.export(format="onnx", imgsz=640)
        
        # Ensure static directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Move exported file to static/yolov8n-pose.onnx
        if os.path.exists(exported_file):
            shutil.move(exported_file, target_path)
            print(f"Successfully exported and saved YOLOv8-Pose ONNX model to {target_path}")
            return target_path
        else:
            print(f"Export completed but output file {exported_file} was not found.")
            return None
    except Exception as e:
        print(f"Failed to auto-download/export YOLOv8-Pose ONNX: {e}")
        return None

if __name__ == "__main__":
    ensure_yolo_pose_onnx()
