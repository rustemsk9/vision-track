import os
import shutil

def ensure_yolo11n_onnx():
    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static/yolo11n.onnx"))
    
    if os.path.exists(target_path):
        print(f"yolo11n.onnx already exists at: {target_path}")
        return target_path

    print("yolo11n.onnx not found in static/. Downloading and exporting YOLOv11-nano...")
    try:
        from ultralytics import YOLO
        
        # Download and load YOLOv11-nano
        model = YOLO("yolo11n.pt")
        
        # Export to ONNX with 640x640 input resolution
        exported_file = model.export(format="onnx", imgsz=640)
        
        # Ensure static directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Move exported file to static/yolo11n.onnx
        if os.path.exists(exported_file):
            shutil.move(exported_file, target_path)
            print(f"Successfully exported and saved YOLOv11-nano ONNX model to {target_path}")
            return target_path
        else:
            print(f"Export completed but output file {exported_file} was not found.")
            return None
    except Exception as e:
        print(f"Failed to auto-download/export YOLOv11-nano ONNX: {e}")
        return None

if __name__ == "__main__":
    ensure_yolo11n_onnx()
