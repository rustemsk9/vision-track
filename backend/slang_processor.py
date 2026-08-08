import torch
import os
try:
    import slangpy as spy
    SLANG_AVAILABLE = True
except ImportError:
    SLANG_AVAILABLE = False
    print("Warning: slangpy not installed. Slang processing will run in CPU fallback mode.")

class SlangHybridProcessor:
    """
    Connects PyTorch GCN outputs to Slang compute shaders for high-fps 
    3D geometric transforms (e.g. skinning, kinematics) via zero-copy GPU tensors.
    """
    def __init__(self, shader_path="shaders/skeleton.slang"):
        self.shader_path = shader_path
        self.module = None
        
        if SLANG_AVAILABLE:
            if not os.path.exists(self.shader_path):
                print(f"Shader not found at {self.shader_path}")
            else:
                try:
                    self.module = spy.loadModule(self.shader_path)
                    print(f"Slang module {self.shader_path} loaded successfully.")
                except Exception as e:
                    print(f"Failed to load Slang module: {e}")
                    
    def process_joints(self, joints_tensor):
        """
        Executes the Slang compute shader on the given joints.
        :param joints_tensor: PyTorch tensor of shape (Batch, Num_Nodes, 3) on CUDA.
        :return: Transformed PyTorch tensor of the same shape.
        """
        if not SLANG_AVAILABLE or self.module is None:
            # Fallback: Just return the input (CPU mock)
            return joints_tensor
            
        # Ensure tensor is contiguous and on CUDA
        if not joints_tensor.is_cuda:
            joints_tensor = joints_tensor.cuda()
            
        joints_tensor = joints_tensor.contiguous()
        
        # Flatten batch and nodes for the 1D compute shader dispatch
        flat_shape = (-1, 3)
        input_flat = joints_tensor.view(flat_shape)
        
        # Allocate output tensor (zero-copy buffer for Slang to write into)
        output_flat = torch.zeros_like(input_flat, device='cuda')
        
        # Calculate dispatch size (blocks of 64 threads)
        num_items = input_flat.shape[0]
        dispatch_size = (num_items // 64) + 1
        
        # Dispatch the shader
        self.module.processJoints(
            inputJoints=input_flat,
            outputTransforms=output_flat
        ).launch(dispatch_size=dispatch_size)
        
        # Reshape back to original dimensions
        output_tensor = output_flat.view(joints_tensor.shape)
        return output_tensor

if __name__ == "__main__":
    # Test script for Slang Integration
    processor = SlangHybridProcessor()
    
    # Dummy GCN output: 1 batch, 17 joints, 3D coordinates
    dummy_input = torch.randn(1, 17, 3)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        
    print("Input Tensor Shape:", dummy_input.shape)
    output = processor.process_joints(dummy_input)
    print("Output Tensor Shape:", output.shape)
    print("Slang Integration Test Complete.")
