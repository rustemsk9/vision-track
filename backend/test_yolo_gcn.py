import os
import cv2
import torch
import numpy as np
from model_gcn import SemanticGCNLifter

def test_yolo_gcn_end_to_end(video_path="test1.mp4"):
    print("=== Testing YOLO Pose -> 3D GCN End-to-End Pipeline ===")

    # 1. Load 3D GCN Model
    weights_path = os.path.abspath("3d_lifter_gcn_pro.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.abspath("backend/3d_lifter_gcn_pro.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    
    gcn_model = SemanticGCNLifter(num_nodes=17, in_channels=5, hidden_channels=128, out_channels=4).to(device)
    if os.path.exists(weights_path):
        gcn_model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded GCN weights successfully!")
    else:
        print("ERROR: GCN weights not found!")
        return

    gcn_model.eval()

    # 2. Try loading YOLO Pose
    try:
        from ultralytics import YOLO
        yolo_pose = YOLO("yolov8n-pose.pt")
        print("Loaded YOLOv8-pose model successfully!")
    except Exception as e:
        print(f"Failed to load YOLO pose: {e}")
        return

    # Static adjacency matrix
    bone_pairs = [
        [0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8], [8,9],
        [9,10], [9,11], [11,12], [12,13], [9,14], [14,15], [15,16]
    ]
    adj = np.eye(17, dtype=np.float32)
    for i, j in bone_pairs:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj_tensor = torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(device)

    # COCO 17 Keypoints to 17 Symmetrical Joint Order
    coco_map = [0, 12, 14, 16, 11, 13, 15, 5, 6, 0, 0, 5, 7, 9, 6, 8, 10]

    # 3. Process Video / Webcam Frames
    vid_path = os.path.abspath(video_path)
    if not os.path.exists(vid_path):
        vid_path = os.path.abspath(os.path.join("..", video_path))

    print(f"Reading video file: {vid_path}")
    cap = cv2.VideoCapture(vid_path)

    frame_idx = 0
    with torch.no_grad():
        while frame_idx < 10:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            results = yolo_pose(frame, verbose=False)
            
            if len(results) > 0 and len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0].cpu().numpy() # Shape [17, 3] -> x, y, conf
                
                # Normalize 2D keypoints to [-1.5, 1.5]
                h_img, w_img = frame.shape[:0], frame.shape[1]
                nodes_5d = np.zeros((17, 5), dtype=np.float32)
                
                for k in range(17):
                    coco_i = coco_map[k]
                    kx, ky, kc = kpts[coco_i]
                    nodes_5d[k, 0] = (kx / w_img) * 3.0 - 1.5
                    nodes_5d[k, 1] = -((ky / h_img) * 3.0 - 1.5)
                    nodes_5d[k, 2] = 10.0 # scale
                    nodes_5d[k, 3] = 10.0 # r_laplacian
                    nodes_5d[k, 4] = kc   # conf
                    
                nodes_tensor = torch.tensor(nodes_5d, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Predict 3D Joints
                pred_3d = gcn_model(nodes_tensor, adj_tensor)[0, :, :3].cpu().numpy()
                
                # Relative offsets around Pelvis root (joint 0)
                root_xyz = pred_3d[0]
                rel_3d = (pred_3d - root_xyz) * 1.8
                
                x_min, x_max = rel_3d[:, 0].min(), rel_3d[:, 0].max()
                y_min, y_max = rel_3d[:, 1].min(), rel_3d[:, 1].max()
                z_min, z_max = rel_3d[:, 2].min(), rel_3d[:, 2].max()
                
                print(f"Frame {frame_idx:02d} | 2D Keypoint[0]: ({nodes_5d[0,0]:.2f}, {nodes_5d[0,1]:.2f}) | 3D Bounds -> X:[{x_min:.2f}, {x_max:.2f}] Y:[{y_min:.2f}, {y_max:.2f}] Z:[{z_min:.2f}, {z_max:.2f}]")
            else:
                print(f"Frame {frame_idx:02d} | No person detected")

    cap.release()
    print("=== End-to-End Test Complete ===")

if __name__ == "__main__":
    test_yolo_gcn_end_to_end()
