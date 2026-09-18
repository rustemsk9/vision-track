# Only for 2d bounding boxes
import os
import platform
import hashlib
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO

def get_current_machine_info():
    """Generates hardware and platform telemetry signature for current machine."""
    if torch.cuda.is_available():
        accelerator = f"cuda:{torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = f"cpu:{platform.processor() or platform.machine()}"

    info = {
        "node": platform.node(),
        "system": platform.system(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "accelerator": accelerator
    }
    raw_sig = f"{info['node']}-{info['platform']}-{info['architecture']}-{info['accelerator']}"
    info["signature"] = hashlib.sha256(raw_sig.encode()).hexdigest()[:16]
    return info

def ensure_checkpoint_for_current_machine(checkpoints_dir="models/checkpoints"):
    """
    Validates that optimized checkpoints exist and match the current machine signature.
    If a new machine or hardware configuration is detected, triggers automated re-optimization.
    """
    ckpt_dir = Path(checkpoints_dir)
    best_pt = ckpt_dir / "best.pt"
    config_yaml = ckpt_dir / "config.yaml"

    curr_machine = get_current_machine_info()
    needs_optimization = False
    reason = ""

    if not best_pt.exists():
        needs_optimization = True
        reason = f"Checkpoint '{best_pt}' not found."
    elif not config_yaml.exists():
        needs_optimization = True
        reason = f"Configuration '{config_yaml}' not found."
    else:
        try:
            with open(config_yaml, "r") as f:
                cfg = yaml.safe_load(f) or {}
            saved_machine = cfg.get("machine", {})
            saved_sig = saved_machine.get("signature")
            if not saved_sig:
                # Stamp current machine signature into existing config
                cfg["machine"] = curr_machine
                with open(config_yaml, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            elif saved_sig != curr_machine["signature"]:
                needs_optimization = True
                prev_desc = f"{saved_machine.get('node', 'unknown')} ({saved_machine.get('accelerator', 'unknown')})"
                curr_desc = f"{curr_machine['node']} ({curr_machine['accelerator']})"
                reason = f"New machine detected! Previous: {prev_desc} -> Current: {curr_desc}."
        except Exception as e:
            needs_optimization = True
            reason = f"Unable to parse checkpoint config ({e})."

    if needs_optimization:
        print(f"[VisionTrack] ⚙️  {reason}")
        print("[VisionTrack] 🚀 Automatically starting model re-optimization & benchmarking for this machine...")
        try:
            from tools.optimize_model import run_optimization
            run_optimization()
            print("[VisionTrack] ✅ Checkpoint successfully calibrated and saved for current machine.")
        except Exception as e:
            print(f"[VisionTrack] ⚠️  Auto-optimization warning: {e}. Falling back to baseline weights.")

class PersonDetector:
    def __init__(self, model_path=None):
        # Check machine match and calibrate checkpoint if on a new machine
        if model_path is None:
            ensure_checkpoint_for_current_machine()
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
