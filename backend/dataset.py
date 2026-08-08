import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from mat_extract import MATExtractor

class VisionTrackDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        PyTorch Dataset for VisionTrack SemanticGCN.
        :param data_dir: Path to the root `training_data_output` folder.
        """
        self.data_dir = data_dir
        self.samples = []
        self.transform = transform
        self.mat_extractor = MATExtractor(num_nodes=17)
        
        # Parse the dataset directory
        self._parse_directory()

    def _parse_directory(self):
        """Recursively parses the training data directory for joints.jsonl and masks."""
        for root, dirs, files in os.walk(self.data_dir):
            if 'joints.jsonl' in files:
                joints_file = os.path.join(root, 'joints.jsonl')
                with open(joints_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        frame_idx = data['frame']
                        joints_dict = data['joints']
                        
                        mask_filename = f"mask_{frame_idx:04d}.png"
                        mask_path = os.path.join(root, mask_filename)
                        
                        if os.path.exists(mask_path):
                            self.samples.append({
                                'mask_path': mask_path,
                                'joints': joints_dict
                            })
                            
        print(f"Loaded {len(self.samples)} valid samples from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mask_path = sample['mask_path']
        joints_dict = sample['joints']
        
        # 1. Input: Extract 2D Graph (Nodes and Adjacency) from Mask
        nodes_2d, adj_matrix = self.mat_extractor.extract_skeleton(mask_path)
        
        if nodes_2d is None:
            # Fallback for empty mask (return zero tensors)
            nodes_2d = np.zeros((17, 2), dtype=np.float32)
            adj_matrix = np.zeros((17, 17), dtype=np.float32)
            
        # The SemanticGCNLifter expects 5 input channels (x, y, r_left, r_right, v)
        # We append rudimentary values for r_left, r_right (distance to boundary) and visibility (v).
        # For now, we pad with 10.0 for radii and 1.0 for visibility.
        num_nodes = nodes_2d.shape[0]
        padding = np.zeros((num_nodes, 3), dtype=np.float32)
        padding[:, 0] = 10.0  # r_left
        padding[:, 1] = 10.0  # r_right
        padding[:, 2] = 1.0   # v (visibility)
        
        nodes_5d = np.hstack((nodes_2d, padding))
        
        nodes_tensor = torch.tensor(nodes_5d, dtype=torch.float32) # Shape: (17, 5)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32) # Shape: (17, 17)
        
        # 2. Target: Extract 3D Joints (Exactly 17 Symmetrical Joints)
        target_joints = []
        core_joint_names = [
            'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 
            'L_Hip', 'L_Knee', 'L_Ankle', 
            'Spine1', 'Spine2', 'Neck', 'Head', 
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 
            'R_Shoulder', 'R_Elbow', 'R_Wrist'
        ]
        
        for name in core_joint_names:
            if name in joints_dict:
                target_joints.append(joints_dict[name])
            else:
                target_joints.append([0.0, 0.0, 0.0]) # Fallback if missing
                
        target_tensor = torch.tensor(target_joints, dtype=torch.float32) # Shape: (17, 3)
        
        return nodes_tensor, adj_tensor, target_tensor

if __name__ == "__main__":
    # Test block
    print("PyTorch Pro Dataset Loader Ready.")
