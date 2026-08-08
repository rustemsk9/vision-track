# VISION-TRACK AI MEMORY BACKUP

> **Instructions for Tomorrow:**
> Just tell the AI: *'Read the AI_MEMORY_BACKUP.md file in the project folder to restore your context, then let's continue with Phase 2.'*

## --- IMPLEMENTATION_PLAN.MD ---

# Phase 2: GCN Training & Medial Axis Transform Pipeline

Now that we have successfully generated the synthetic dataset (RGB masks + `joints.jsonl` ground truth), the next major milestone is building the PyTorch pipeline to train our 3D Graph Convolutional Network (GCN).

## Open Questions
1. **Medial Axis Transform (MAT) Resolution:** To keep the GCN extremely fast for webcam inference (60+ FPS), we should extract a sparse 2D graph from the YOLO/Blender masks using MAT. Do you want to process the MAT in Python using OpenCV/Scikit-Image for training, and later port it to WASM/JS for the browser frontend?
2. **GCN Architecture:** A standard Spatial-Temporal Graph Convolutional Network (ST-GCN) is ideal for this. Should we stick to a lightweight 2-3 block GCN to ensure it runs flawlessly via ONNX on the browser edge?

## Proposed Changes

### 1. PyTorch Dataset Loader
*   Create a custom PyTorch `Dataset` in `backend/train_gcn.py`.
*   This loader will parse the `training_data_output` folder.
*   For each frame, it will read `joints.jsonl` and extract the 17 core SMPL joints as a flat target tensor of shape `(17, 3)`.

### 2. Medial Axis Transform (MAT) Pre-Processing
*   We cannot feed raw 640x480 pixels directly into a GCN (that requires a CNN). A GCN requires a *graph* (Nodes and Edges).
*   We will implement a MAT extraction function using OpenCV/Scikit-image.
*   **Workflow:** Mask `.png` -> Medial Axis Skeleton -> Sample $N$ key points along the skeleton -> Generate Adjacency Matrix -> Feed into GCN.

### 3. Lightweight 3D GCN Model
*   Design a PyTorch model (`3d_lifter_gcn.py`) utilizing `torch_geometric` or standard `torch.nn.Linear` approximations of graph convolutions.
*   **Input:** 2D Graph Nodes $(X, Y)$ extracted from the MAT.
*   **Output:** 3D Joint Coordinates $(X, Y, Z)$ for 17 joints.
*   **Loss Function:** Mean Squared Error (MSE) or L1 Loss against the `joints.jsonl` ground truth.

### 4. ONNX Export
*   Once the model is trained on the synthetic dataset, we will trace and export it as `3d_lifter_gcn.onnx`.
*   We will copy this `.onnx` file into `static/models/` so our frontend Three.js app can finally replace the simulated math loop with real, edge-computed 3D inference!

## Verification Plan
*   Run the MAT extractor on a sample Blender PNG and plot the extracted 2D skeleton to verify it aligns with the silhouette.
*   Train the GCN for 10-20 epochs and verify the loss decreases.
*   Run the exported ONNX model in a test script to ensure inference takes <10ms.


## --- TASK.MD ---

# VisionTrack Generator Batch Enhancements Tasks

- `[x]` 1. Update `generator_ops.py` UI and Properties (Switch to directory selection)
- `[x]` 2. Implement Dynamic Rig Detection (Search for ARMATURE with 'Pelvis')
- `[x]` 3. Update Camera & Posture Logic (Height 1.2m, 640x480, Random Z-Rotation)
- `[x]` 4. Optimize Mask Export (8-bit Grayscale, 100% compression)
- `[x]` 5. Implement Batch Processing Loop & Logging (Iterate .npz files, generate subfolders)


## --- WALKTHROUGH.MD ---

# VisionTrack Generator: Batch Processing Update

I have completely rewritten the Blender add-on to act as a highly optimized Batch Processor for your dataset generation! Here is a summary of the powerful new features you can use right now.

## 1. Batch `.npz` Folder Selection
When you click **Batch Generate Data** in the VisionTrack sidebar, the file browser will now ask you to select a *Folder* rather than a file.
*   Once selected, the script automatically finds every single `.npz` file inside that folder.
*   It automatically iterates through them, loads the animation, renders the masks, exports the `joints.jsonl` data, and creates a cleanly named sub-folder for every sequence inside `training_data_output/`.

## 2. Dynamic Rig & Mesh Detection
You no longer need to worry about manually renaming your imported rigs! 
*   The script scans your scene and looks for *any* Armature that contains a `Pelvis` bone. Whether you use `SMPL-male`, `SMPL-female`, or `SMPL-neutral`, it will automatically detect and bind to it.

## 3. Data Augmentation & Framing
To properly train the GCN to understand different heights and perspectives:
*   **Camera Adjustments:** The camera is now perfectly positioned 1.2m off the ground (desk height), aiming squarely down the Y-axis.
*   **Randomized View Angles:** Right before rendering an `.npz` sequence, the script randomly rotates the Z-axis of the SMPL rig. This means your GCN will learn to track bodies that are facing the camera, turning away, or walking sideways!

## 4. Extreme Output Optimization
To solve your storage concerns regarding the 1.4MB image sizes:
*   The output resolution is now hardcoded to `640x480` (Standard Webcam Resolution).
*   The output format is forced to 8-bit Grayscale (BW) instead of RGBA.
*   PNG compression is bumped up to 100%.
*   > [!TIP]
    > **Storage Savings**
    > These changes should shrink your mask file sizes from `1.4MB` down to approximately `10KB - 50KB` per frame, saving massive amounts of SSD space!

## 5. Built-in Logger
Because processing an entire folder might take a few minutes, the add-on now generates a `visiontrack_generator.log` file directly inside your `training_data_output` folder. You can open this file in VSCode to see exactly which sequence it is currently processing, how many frames it rendered, and if there were any errors along the way.


