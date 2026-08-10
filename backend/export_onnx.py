import torch
import os
from model_gcn import SemanticGCNLifter

def export_to_onnx():
    # 1. Initialize model and load weights
    model = SemanticGCNLifter(num_nodes=17, in_channels=5, hidden_channels=128, out_channels=4)
    weights_path = "3d_lifter_gcn_pro.pth"
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Exporting untrained model.")
        
    model.eval()
    
    # 2. Create dummy inputs matching the expected shape
    # Batch size 1, 17 nodes, 5 channels (X, Y, scale, Laplacian radius, visibility)
    dummy_nodes = torch.randn(1, 17, 5, dtype=torch.float32)
    # Batch size 1, 17x17 Adjacency matrix
    dummy_adj = torch.ones(1, 17, 17, dtype=torch.float32)
    
    # 3. Export configuration
    output_path = os.path.abspath("../static/models/3d_lifter_gcn.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Remove any existing export files to prevent PyTorch from writing external data sidecars
    for path in [output_path, output_path + ".data"]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
                
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        (dummy_nodes, dummy_adj), 
        output_path,
        export_params=True,
        opset_version=18,          # Standard ONNX opset 18
        do_constant_folding=True,  # Optimize constants
        input_names=['input_nodes', 'input_adj'],
        output_names=['output_joints'],
        dynamic_axes={
            'input_nodes': {0: 'batch_size'},
            'input_adj': {0: 'batch_size'},
            'output_joints': {0: 'batch_size'}
        },
        dynamo=False
    )
    
    print("ONNX Export complete! The model is ready for Three.js / WebGL deployment.")

if __name__ == "__main__":
    export_to_onnx()
