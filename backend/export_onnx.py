import torch
import os
from model_gcn import LifterGCN

def export_to_onnx():
    # 1. Initialize model and load weights
    model = LifterGCN(num_nodes=17, in_dim=2, hidden_dim=64, out_dim=3)
    weights_path = "3d_lifter_gcn.pth"
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Exporting untrained model.")
        
    model.eval()
    
    # 2. Create dummy inputs matching the expected shape
    # Batch size 1, 17 nodes, 2D coordinates
    dummy_nodes = torch.randn(1, 17, 2, dtype=torch.float32)
    # Batch size 1, 17x17 Adjacency matrix
    dummy_adj = torch.ones(1, 17, 17, dtype=torch.float32)
    
    # 3. Export configuration
    output_path = os.path.abspath("../static/models/3d_lifter_gcn.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        (dummy_nodes, dummy_adj), 
        output_path,
        export_params=True,
        opset_version=14,          # Standard ONNX opset
        do_constant_folding=True,  # Optimize constants
        input_names=['input_nodes', 'input_adj'],
        output_names=['output_joints'],
        dynamic_axes={
            'input_nodes': {0: 'batch_size'},
            'input_adj': {0: 'batch_size'},
            'output_joints': {0: 'batch_size'}
        }
    )
    
    print("ONNX Export complete! The model is ready for Three.js / WebGL deployment.")

if __name__ == "__main__":
    export_to_onnx()
