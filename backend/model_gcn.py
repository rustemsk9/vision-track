import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticGraphConv(nn.Module):
    """
    Upgraded Semantic Graph Convolutional Layer.
    Computes: H' = sigma( ( (A + I) * M ) * H * W )
    """
    def __init__(self, in_features, out_features, num_nodes=17):
        super(SemanticGraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # The "Pro" Upgrade: Learnable Semantic Mask (M)
        self.M = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.M, 1.0) # Initialize to 1.0 for standard GCN flow

    def forward(self, x, adj):
        """
        x: (Batch, Num_Nodes, In_Features)
        adj: Static Adjacency Matrix (Batch, Num_Nodes, Num_Nodes)
        """
        # Create self-loops: A + I
        I = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        A_hat = adj + I

        # Symmetric degree normalization: D^(-1/2) * A_hat * D^(-1/2)
        deg = torch.sum(A_hat, dim=-1)
        deg_inv_sqrt = torch.pow(deg + 1e-5, -0.5)
        D_inv_sqrt = I * deg_inv_sqrt.unsqueeze(-1) # ONNX-compatible diagonal matrix
        norm_adj = torch.matmul(torch.matmul(D_inv_sqrt, A_hat), D_inv_sqrt)
        
        # Apply the semantic learnable mask (element-wise multiplication)
        semantic_adj = norm_adj * self.M
        
        # H * W
        support = torch.matmul(x, self.weight)
        
        # A_semantic * (H * W)
        out = torch.matmul(semantic_adj, support) + self.bias
        return out


class SemanticGCNLifter(nn.Module):
    """
    Node D: The 3-Layer 3D Lifter Network (Direct 3D Spatial Coordinate Regression: X, Y, Z)
    """
    def __init__(self, num_nodes=17, in_channels=5, hidden_channels=128, out_channels=3):
        super(SemanticGCNLifter, self).__init__()
        
        # Layer 1: Input (5) -> Hidden
        self.gcn1 = SemanticGraphConv(in_channels, hidden_channels, num_nodes)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # Layer 2: Hidden -> Hidden
        self.gcn2 = SemanticGraphConv(hidden_channels, hidden_channels, num_nodes)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Layer 3: Hidden -> Output (3 channels: X, Y, Z)
        self.gcn3 = SemanticGraphConv(hidden_channels, out_channels, num_nodes)
        
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x, adj):
        # x shape: (Batch, 17, 5)
        
        # Block 1
        x = self.gcn1(x, adj)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2) # BatchNorm expects (N, C, L)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        # Block 2
        x = self.gcn2(x, adj)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        # Block 3 (No activation or dropout on final metric regression)
        out = self.gcn3(x, adj)
        
        return out

if __name__ == "__main__":
    # Test model
    model = SemanticGCNLifter()
    dummy_x = torch.randn(1, 17, 5)
    dummy_adj = torch.ones(1, 17, 17)
    output = model(dummy_x, dummy_adj)
    print(f"Model Output Shape: {output.shape} (Expected: 1, 17, 4)")
