import os
import torch
import cv2
import numpy as np
from model_gcn import SemanticGCNLifter
from mat_extract import MATExtractor

def test_inference_on_video_or_image(video_path="test1.mp4"):
    print("=== Testing 3D GCN Pipeline on Local Video / Image ===")
    
    # 1. Load Model
    weights_path = os.path.abspath("3d_lifter_gcn_pro.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.abspath("backend/3d_lifter_gcn_pro.pth")
        
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Loading GCN weights from: {weights_path}")
    
    model = SemanticGCNLifter(num_nodes=17, in_channels=5, hidden_channels=128, out_channels=4).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded successfully!")
    else:
        print("ERROR: Weights file not found!")
        return

    model.eval()

    # 2. Initialize MAT Extractor
    extractor = MATExtractor(num_nodes=17)
    
    # 3. Open Video / Image
    vid_full_path = os.path.abspath(video_path)
    if not os.path.exists(vid_full_path):
        vid_full_path = os.path.abspath(os.path.join("..", video_path))
        
    print(f"Processing media: {vid_full_path}")
    cap = cv2.VideoCapture(vid_full_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {vid_full_path}")
        return
        
    frame_count = 0
    with torch.no_grad():
        while frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Convert frame to binary mask representation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # Save temporary mask
            tmp_mask_path = "tmp_test_mask.png"
            cv2.imwrite(tmp_mask_path, mask)
            
            nodes_2d, adj_matrix = extractor.extract_skeleton(tmp_mask_path)
            if os.path.exists(tmp_mask_path):
                os.remove(tmp_mask_path)
                
            if nodes_2d is None:
                print(f"Frame {frame_count}: No silhouette detected.")
                continue
                
            num_nodes = nodes_2d.shape[0]
            padding = np.zeros((num_nodes, 3), dtype=np.float32)
            padding[:, 0] = 10.0
            padding[:, 1] = 10.0
            padding[:, 2] = 1.0
            
            nodes_5d = np.hstack((nodes_2d, padding))
            nodes_tensor = torch.tensor(nodes_5d, dtype=torch.float32).unsqueeze(0).to(device) # (1, 17, 5)
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0).to(device)   # (1, 17, 17)
            
            # Run GCN inference
            pred_3d = model(nodes_tensor, adj_tensor) # (1, 17, 4) -> X, Y, Z, Sigma_Z
            pred_xyz = pred_3d[0, :, :3].cpu().numpy()
            
            z_min, z_max = pred_xyz[:, 2].min(), pred_xyz[:, 2].max()
            x_min, x_max = pred_xyz[:, 0].min(), pred_xyz[:, 0].max()
            y_min, y_max = pred_xyz[:, 1].min(), pred_xyz[:, 1].max()
            
            print(f"Frame {frame_count:02d} | 3D Bounds -> X:[{x_min:.2f}, {x_max:.2f}] Y:[{y_min:.2f}, {y_max:.2f}] Z:[{z_min:.2f}, {z_max:.2f}]")

    cap.release()
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_inference_on_video_or_image()
