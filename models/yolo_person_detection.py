import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLO model for person detection
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
