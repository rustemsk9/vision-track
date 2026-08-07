Phase 2: GCN Training & Medial Axis Transform Pipeline
Now that we have successfully generated the synthetic dataset (RGB masks + joints.jsonl ground truth), the next major milestone is building the PyTorch pipeline to train our 3D Graph Convolutional Network (GCN).

Open Questions
Medial Axis Transform (MAT) Resolution: To keep the GCN extremely fast for webcam inference (60+ FPS), we should extract a sparse 2D graph from the YOLO/Blender masks using MAT. Do you want to process the MAT in Python using OpenCV/Scikit-Image for training, and later port it to WASM/JS for the browser frontend?
GCN Architecture: A standard Spatial-Temporal Graph Convolutional Network (ST-GCN) is ideal for this. Should we stick to a lightweight 2-3 block GCN to ensure it runs flawlessly via ONNX on the browser edge?
Proposed Changes
1. PyTorch Dataset Loader
Create a custom PyTorch Dataset in backend/train_gcn.py.
This loader will parse the training_data_output folder.
For each frame, it will read joints.jsonl and extract the 17 core SMPL joints as a flat target tensor of shape (17, 3).
2. Medial Axis Transform (MAT) Pre-Processing
We cannot feed raw 640x480 pixels directly into a GCN (that requires a CNN). A GCN requires a graph (Nodes and Edges).
We will implement a MAT extraction function using OpenCV/Scikit-image.
Workflow: Mask .png -> Medial Axis Skeleton -> Sample N key points along the skeleton -> Generate Adjacency Matrix -> Feed into GCN.
3. Lightweight 3D GCN Model
Design a PyTorch model (3d_lifter_gcn.py) utilizing torch_geometric or standard torch.nn.Linear approximations of graph convolutions.
Input: 2D Graph Nodes (X,Y) extracted from the MAT.
Output: 3D Joint Coordinates (X,Y,Z) for 17 joints.
Loss Function: Mean Squared Error (MSE) or L1 Loss against the joints.jsonl ground truth.
4. ONNX Export
Once the model is trained on the synthetic dataset, we will trace and export it as 3d_lifter_gcn.onnx.
We will copy this .onnx file into static/models/ so our frontend Three.js app can finally replace the simulated math loop with real, edge-computed 3D inference!
Verification Plan
Run the MAT extractor on a sample Blender PNG and plot the extracted 2D skeleton to verify it aligns with the silhouette.
Train the GCN for 10-20 epochs and verify the loss decreases.
Run the exported ONNX model in a test script to ensure inference takes <10ms.