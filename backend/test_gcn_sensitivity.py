import os
import torch
import numpy as np
from model_gcn import SemanticGCNLifter

def test_gcn_pose_sensitivity():
    print("=== Testing GCN Model Sensitivity to Different 2D Poses ===")

    weights_path = os.path.abspath("3d_lifter_gcn_pro.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.abspath("backend/3d_lifter_gcn_pro.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")
    
    model = SemanticGCNLifter(num_nodes=17, in_channels=5, hidden_channels=128, out_channels=4).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded successfully!")
    else:
        print("ERROR: Weights file not found!")
        return

    model.eval()

    # Bone adjacency matrix
    bone_pairs = [
        [0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8], [8,9],
        [9,10], [9,11], [11,12], [12,13], [9,14], [14,15], [15,16]
    ]
    adj = np.eye(17, dtype=np.float32)
    for i, j in bone_pairs:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj_tensor = torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(device)

    # Pose 1: Standing Straight
    nodes_p1 = np.zeros((17, 5), dtype=np.float32)
    for i in range(17):
        nodes_p1[i, 0] = 0.0
        nodes_p1[i, 1] = (i / 16.0) * 2.0 - 1.0
        nodes_p1[i, 2] = 10.0
        nodes_p1[i, 3] = 10.0
        nodes_p1[i, 4] = 1.0

    # Pose 2: Arms Raised (Wrists nodes 13 & 16 raised up to y = +1.5)
    nodes_p2 = nodes_p1.copy()
    nodes_p2[13, 0] = -1.2 # Left Wrist left & up
    nodes_p2[13, 1] = 1.5
    nodes_p2[16, 0] = 1.2  # Right Wrist right & up
    nodes_p2[16, 1] = 1.5

    # Pose 3: Leaning Right (x shifted by +0.8)
    nodes_p3 = nodes_p1.copy()
    nodes_p3[:, 0] += 0.8

    poses = [("Standing Straight", nodes_p1), ("Arms Raised Overhead", nodes_p2), ("Leaning Right", nodes_p3)]

    with torch.no_grad():
        for name, nodes_arr in poses:
            tensor_in = torch.tensor(nodes_arr, dtype=torch.float32).unsqueeze(0).to(device)
            output_3d = model(tensor_in, adj_tensor)[0, :, :3].cpu().numpy() # [17, 3]
            
            root = output_3d[0]
            rel_3d = output_3d - root
            
            x_range = (rel_3d[:, 0].min(), rel_3d[:, 0].max())
            y_range = (rel_3d[:, 1].min(), rel_3d[:, 1].max())
            z_range = (rel_3d[:, 2].min(), rel_3d[:, 2].max())
            
            print(f"\n--- Pose: {name} ---")
            print(f"Root Pelvis 3D (X, Y, Z): ({root[0]:.2f}, {root[1]:.2f}, {root[2]:.2f})")
            print(f"Relative 3D Bounds -> X:[{x_range[0]:.2f}, {x_range[1]:.2f}] Y:[{y_range[0]:.2f}, {y_range[1]:.2f}] Z:[{z_range[0]:.2f}, {z_range[1]:.2f}]")
            print(f"Left Wrist 3D:  ({rel_3d[13,0]:.2f}, {rel_3d[13,1]:.2f}, {rel_3d[13,2]:.2f})")
            print(f"Right Wrist 3D: ({rel_3d[16,0]:.2f}, {rel_3d[16,1]:.2f}, {rel_3d[16,2]:.2f})")

if __name__ == "__main__":
    test_gcn_pose_sensitivity()
