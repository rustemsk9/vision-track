# Only for 2d bounding boxes
import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path=None):
        import os
        # Load optimized pruned checkpoint if available, else fallback to yolov8n.pt
        if model_path is None:
            if os.path.exists("models/checkpoints/best.pt"):
                model_path = "models/checkpoints/best.pt"
            else:
                model_path = "yolov8n.pt"
        self.model = YOLO(model_path)
        
        # Detect Hardware
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
    
    def predict(self, frame, conf=0.5, **kwargs):
        # We filter for class 0 (person in COCO) and run on the optimal device
        results = self.model(frame, classes=[0], conf=conf, device=self.device, **kwargs)
        return results[0]
