import cv2
import numpy as np
from skimage.morphology import medial_axis
import json
import os

class MATExtractor:
    def __init__(self, num_nodes=17):
        """
        Initializes the Medial Axis Transform extractor.
        :param num_nodes: Number of key points to sample along the medial axis skeleton.
        """
        self.num_nodes = num_nodes

    def extract_skeleton(self, mask_path):
        """
        Extracts a sparse 2D graph from a binary mask using Medial Axis Transform.
        :param mask_path: Path to the binary mask image.
        :return: (nodes, adjacency_matrix) or (None, None) if extraction fails.
        """
        # 1. Read mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
            
        # 2. Binarize
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 3. Simulate YOLO Bounding Box Crop
        # Find the active pixels (the human silhouette)
        y_indices, x_indices = np.where(binary > 0)
        if len(y_indices) == 0:
            return None, None
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # Crop the mask exactly to the bounding box, just like YOLO does
        cropped_mask = binary[y_min:y_max+1, x_min:x_max+1]
        
        # Resize to a standard YOLO output size (e.g., 256x256) to ensure scale invariance
        target_size = (256, 256)
        resized_mask = cv2.resize(cropped_mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # 4. Compute Medial Axis Transform on the YOLO-cropped mask
        # Convert to boolean for skimage
        bool_mask = resized_mask > 0
        skeleton, distance = medial_axis(bool_mask, return_distance=True)
        
        # Find coordinates of the skeleton pixels
        skeleton_coords = np.column_stack(np.where(skeleton))
        
        if len(skeleton_coords) == 0:
            return None, None
            
        # 4. Subsample the skeleton to get exactly `num_nodes` points.
        # A robust way is to use KMeans clustering on the skeleton pixels 
        # to find representative centers (nodes).
        from sklearn.cluster import KMeans
        if len(skeleton_coords) < self.num_nodes:
            # Fallback if too few pixels (rare for a human mask)
            return None, None
            
        kmeans = KMeans(n_clusters=self.num_nodes, random_state=42, n_init='auto')
        kmeans.fit(skeleton_coords)
        nodes = kmeans.cluster_centers_  # shape: (num_nodes, 2)
        
        # Sort nodes by Y coordinate (roughly head to toe) for consistency
        nodes = nodes[nodes[:, 0].argsort()]
        
        # 5. Generate Adjacency Matrix (using simple K-Nearest Neighbors for now, 
        # or distance threshold, representing bone connections)
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        from scipy.spatial import distance_matrix
        dist_mat = distance_matrix(nodes, nodes)
        
        # Connect each node to its 2 nearest neighbors to form a graph
        for i in range(self.num_nodes):
            nearest_idx = np.argsort(dist_mat[i])[1:3] # skip self (index 0)
            for idx in nearest_idx:
                adj_matrix[i, idx] = 1
                adj_matrix[idx, i] = 1 # Make it symmetric

        return nodes, adj_matrix

if __name__ == "__main__":
    # Test block
    print("MAT Extractor Ready.")
