"""
Data loading utilities for VisionTrack
Provides robust stream loaders for single-stream, multi-stream, and image datasets.
"""

import os
import cv2
from pathlib import Path
from typing import Generator, Tuple, Optional, List, Dict, Any

class VideoStreamLoader:
    """Loads frames from a video file, stream, or camera device."""
    def __init__(self, source: str, target_size: Optional[Tuple[int, int]] = None):
        self.source = source
        self.target_size = target_size
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2)
        }

    def __iter__(self) -> Generator[Tuple[int, float, Any], None, None]:
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            pts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pts <= 0.001 and frame_idx > 5:
                pts = frame_idx / self.fps
                
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)
                
            yield frame_idx, pts, frame
            frame_idx += 1

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ImageDatasetLoader:
    """Loads image datasets and corresponding YOLO annotations (.txt)."""
    def __init__(self, image_dir: str, label_dir: Optional[str] = None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts
        ]) if self.image_dir.exists() else []

    def __len__(self) -> int:
        return len(self.image_files)

    def get_item(self, idx: int) -> Tuple[Any, List[Dict[str, float]], str]:
        if idx < 0 or idx >= len(self.image_files):
            raise IndexError("Index out of range")
            
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        
        boxes = []
        if self.label_dir:
            txt_path = self.label_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id, xc, yc, w, h = map(float, parts[:5])
                            boxes.append({
                                "class_id": int(cls_id),
                                "x_center": xc,
                                "y_center": yc,
                                "width": w,
                                "height": h
                            })
        return image, boxes, str(img_path)


class MultiStreamDataLoader:
    """Manages multiple simultaneous video streams with distinct stream IDs."""
    def __init__(self, stream_sources: Dict[str, str]):
        self.loaders = {}
        for stream_id, src in stream_sources.items():
            try:
                self.loaders[stream_id] = VideoStreamLoader(src)
            except Exception as e:
                print(f"[MultiStream] Failed to initialize stream '{stream_id}': {e}")

    def get_active_streams(self) -> List[str]:
        return list(self.loaders.keys())

    def close_all(self):
        for loader in self.loaders.values():
            loader.release()
