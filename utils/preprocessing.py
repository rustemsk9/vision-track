# Only for 2d bounding boxes
"""
Preprocessing and Static Camera Optimization utilities for VisionTrack.
Provides standard YOLO resizing, normalization, letterboxing,
and static camera motion gating.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints.
    Returns: (resized_image, scale_ratio, (pad_w, pad_h))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Converts BGR image to RGB and normalizes pixel values from [0, 255] to [0.0, 1.0]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def preprocess_for_yolo(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Complete YOLO preprocessing pipeline: letterbox -> normalize -> (B, C, H, W) tensor.
    """
    padded, ratio, (dw, dh) = letterbox(frame, target_size, auto=False)
    norm = normalize_frame(padded)
    # Transpose HWC -> CHW and add batch dimension -> (1, C, H, W)
    tensor = np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0)
    return np.ascontiguousarray(tensor), ratio, (dw, dh)


class StaticCameraMotionGate:
    """
    Auto-optimization filter for static/fixed cameras.
    Uses adaptive background subtraction and pixel delta gating to detect whether
    motion occurred in the scene, allowing bypass of heavy YOLO inference on static frames.
    """
    def __init__(self, history: int = 100, var_threshold: float = 25.0, motion_threshold: float = 0.001):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.motion_threshold = motion_threshold  # fraction of pixels moving (e.g. 0.1% of frame)
        self.frame_count = 0

    def should_infer(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluates whether significant motion is present in the current frame.
        Returns (should_run_yolo, motion_ratio).
        Always runs for initial calibration frames.
        """
        self.frame_count += 1
        
        # Downscale for ultra-fast motion estimation (e.g. 320x180)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 180))
        fg_mask = self.bg_subtractor.apply(small)
        
        # Warmup period: first 15 frames always infer to stabilize tracker
        if self.frame_count <= 15:
            return True, 1.0

        moving_pixels = cv2.countNonZero(fg_mask)
        total_pixels = 320 * 180
        motion_ratio = moving_pixels / float(total_pixels)

        # If motion exceeds threshold, run full YOLO inference
        should_run = motion_ratio >= self.motion_threshold
        return should_run, motion_ratio
