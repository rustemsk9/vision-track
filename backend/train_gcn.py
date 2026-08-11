import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

from dataset import VisionTrackDataset
from model_gcn import SemanticGCNLifter

class DirectPose3DLoss(nn.Module):
    """
    Stable 3D Keypoint & Kinematic Bone Consistency Loss (No NLL/Variance)
    """
    def __init__(self, bone_pairs, lambda_bone=0.2):
        super(DirectPose3DLoss, self).__init__()
        self.bone_pairs = bone_pairs
        self.lambda_bone = lambda_bone
        if bone_pairs:
            self.register_buffer('start_idx', torch.tensor([p[0] for p in bone_pairs], dtype=torch.long))
            self.register_buffer('end_idx', torch.tensor([p[1] for p in bone_pairs], dtype=torch.long))

    def forward(self, pred_xyz, target_xyz):
        """
        pred_xyz shape:   (Batch, 17, 3) -> X, Y, Z
        target_xyz shape: (Batch, 17, 3) -> X, Y, Z
        """
        pred_xyz = pred_xyz.contiguous()
        target_xyz = target_xyz.contiguous()

        # 1. MPJPE - Smooth L1 (Huber) for robust keypoint regression
        loss_mpjpe = F.smooth_l1_loss(pred_xyz, target_xyz)

        # 2. Kinematic Bone-Length Consistency
        loss_bone = torch.tensor(0.0, device=pred_xyz.device)
        if self.bone_pairs:
            pred_bones = pred_xyz[:, self.start_idx, :] - pred_xyz[:, self.end_idx, :]
            target_bones = target_xyz[:, self.start_idx, :] - target_xyz[:, self.end_idx, :]
            
            pred_lens = torch.sqrt(torch.sum(pred_bones ** 2, dim=-1) + 1e-8)
            target_lens = torch.sqrt(torch.sum(target_bones ** 2, dim=-1) + 1e-8)
            loss_bone = F.l1_loss(pred_lens, target_lens)

        # Combine terms
        total_loss = loss_mpjpe + (self.lambda_bone * loss_bone)
        return total_loss, loss_mpjpe, loss_bone

def train():
    # Configuration
    data_dir = os.path.abspath("../training_data_output")
    batch_size = 32
    epochs = 20
    learning_rate = 0.001
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # 1. Dataset & DataLoader
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please run the Blender Batch Generator first.")
        return
        
    dataset = VisionTrackDataset(data_dir)
    if len(dataset) == 0:
        print("No training data found. Please run the Blender Batch Generator first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define COCO-style bone connections for the kinematic loss based on our 17 symmetrical joints mapping
    # 0:Pelvis, 1:R_Hip, 2:R_Knee, 3:R_Ankle, 4:L_Hip, 5:L_Knee, 6:L_Ankle, 7:Spine1, 8:Spine2, 9:Neck, 10:Head,
    # 11:L_Shoulder, 12:L_Elbow, 13:L_Wrist, 14:R_Shoulder, 15:R_Elbow, 16:R_Wrist
    bone_pairs = [
        (0,1), (1,2), (2,3),     # Right Leg
        (0,4), (4,5), (5,6),     # Left Leg
        (0,7), (7,8), (8,9),     # Spine
        (9,10),                  # Neck to Head
        (9,11), (11,12), (12,13),# Left Arm
        (9,14), (14,15), (15,16) # Right Arm
    ]
    
    # 2. Initialize Upgraded Model
    # 17 nodes, 5 input channels, 128 hidden, 3 output channels (X, Y, Z)
    model = SemanticGCNLifter(num_nodes=17, in_channels=5, hidden_channels=128, out_channels=3).to(device)
    
    # XLA / Torch Compile optimization for local hardware throughput (CUDA only)
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            print("Compiling model for optimized gradient calculations...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Torch compile failed (expected on some OS/environments): {e}. Proceeding without compilation.")
    
    # 3. Custom Loss & Optimizer
    criterion = DirectPose3DLoss(bone_pairs=bone_pairs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Training Loop
    print("Starting Training...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        epoch_loss = 0.0
        epoch_mpjpe = 0.0
        epoch_bone = 0.0
        
        for batch_idx, (nodes, adj, target) in enumerate(dataloader):
            nodes, adj, target = nodes.to(device), adj.to(device), target.to(device)
            
            optimizer.zero_grad()
            predictions = model(nodes, adj)
            
            # Use the Trinity of Losses
            loss, loss_mpjpe, loss_bone = criterion(predictions, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mpjpe += loss_mpjpe.item()
            epoch_bone += loss_bone.item()
            
        avg_loss = epoch_loss / len(dataloader)
        avg_mpjpe = epoch_mpjpe / len(dataloader)
        avg_bone = epoch_bone / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | MPJPE: {avg_mpjpe:.4f}m | Bone: {avg_bone:.4f}m")
        
    # 5. Save Weights ready for ONNX export
    save_path = "3d_lifter_gcn_pro.pth"
    # If compiled model, extract original state dict
    try:
        state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    except Exception:
        state_dict = model.state_dict()
        
    torch.save(state_dict, save_path)
    print(f"Training Complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train()
