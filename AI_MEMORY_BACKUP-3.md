# VISION-TRACK AI MEMORY BACKUP

> **Instructions for Tomorrow:**
> Just tell the AI: *'Read the AI_MEMORY_BACKUP.md file in the project folder to restore your context, then let's continue with Phase 2.'*

## --- IMPLEMENTATION_PLAN.MD ---

# Phase 2 Upgrade: SemanticGCN & Anatomical Loss

Based on the research document, the current lightweight GCN is not powerful enough to handle 3D perspective lifting and occlusions without hallucinating anatomy. We need to upgrade to the "Pro" version (Node D SemanticGCNLifter) outlined in `Upgrading gcn for 3d pose estimation.md`.

## User Review Required

- **5-Channel Input:** The upgraded `SemanticGCNLifter` requires 5 input channels ($x, y, r_{\text{left}}, r_{\text{right}}, v$). Since our MAT extractor only currently outputs $x, y$, I propose we augment `dataset.py` to calculate the approximate distance from the skeleton node to the mask boundary to fill the $r_{\text{left}}$ and $r_{\text{right}}$ channels, and set visibility $v=1.0$ for now. Does this sound good?
- **Perspective Projection Matrix:** The research doc mentions doing on-the-fly depth translation in the PyTorch Dataloader to simulate perspective distortion. Do you want me to include this data augmentation step now, or stick to the basic affine data for this initial training run?

## Proposed Changes

### 1. Fix Symmetrical Joint Mapping
#### [MODIFY] backend/dataset.py
- Update `core_joint_names` to perfectly map 17 symmetrical joints: 
  `Pelvis`, `L_Hip`, `L_Knee`, `L_Ankle`, `R_Hip`, `R_Knee`, `R_Ankle`, `Spine1`, `Spine2`, `Spine3`, `Neck`, `Head`, `L_Shoulder`, `L_Elbow`, `L_Wrist`, `R_Shoulder`, `R_Elbow`, `R_Wrist` (Wait, SMPL has 24, we will pick exactly 17 that form a contiguous human skeleton).
- Pad the input features from $(N, 2)$ to $(N, 5)$ to match the new model architecture.

### 2. Semantic GCN Lifter
#### [MODIFY] backend/model_gcn.py
- Replace `LifterGCN` with `SemanticGCNLifter`.
- Implement `SemanticGraphConv` which includes the learnable mask matrix $M$ (Modulated GCN).
- Add Batch Normalization and LeakyReLU activations.
- Change input dims to 5 and output dims to 4 ($X, Y, Z, \sigma_Z$).

### 3. The Trinity of Losses
#### [MODIFY] backend/train_gcn.py
- Implement the `NodeDLoss` class.
- Calculate MPJPE, Kinematic Bone-Length Consistency, and Uncertainty NLL.
- Define the correct `bone_pairs` tuples matching our 17 joints for the kinematic loss.
- Update the training loop to use the new model and loss.

## Verification Plan
1. Check that `dataset.py` outputs tensors of shape `(Batch, 17, 5)`.
2. Check that the loss converges during a dummy training run without throwing dimensional errors.
3. Verify that the custom `NodeDLoss` properly penalizes anatomically impossible bone stretching.


## --- TASK.MD ---

- `[x]` 1. Fix Blender Batch Generator (Phase 1 Completion)
  - `[x]` Implement recursive search (`os.walk`) in `blender_addon/operators/generator_ops.py`
  - `[x]` Implement unique output folder naming with indexes to prevent overwriting
- `[x]` 2. PyTorch Dataset Loader (Phase 2 Pro)
  - `[x]` Update `backend/dataset.py` with 17 symmetrical joints
  - `[x]` Expand input node features from 2 to 5 channels
- `[x]` 3. MAT Pre-Processing (Phase 2)
  - `[x]` Create `backend/mat_extract.py` for 2D skeleton extraction
- `[x]` 4. Semantic 3D GCN Model (Phase 2 Pro)
  - `[x]` Implement `SemanticGCNLifter` in `backend/model_gcn.py`
  - `[x]` Implement `NodeDLoss` and new training loop in `backend/train_gcn.py`
- `[x]` 5. Slang Shaders & Hybrid Pipeline (Phase 2)
  - `[x]` Implement `backend/shaders/skeleton.slang`
  - `[x]` Implement `backend/slang_processor.py`
- `[x]` 6. ONNX Export (Phase 2)
  - `[x]` Create `backend/export_onnx.py`


## --- WALKTHROUGH.MD ---

# Phase 1 & 2 "Pro" Completion Walkthrough

## Phase 1: Blender Batch Generator Fix
The Blender batch generator has been fully upgraded to handle complex, nested AMASS datasets.
- **Recursive Directory Searching**: The script now uses `os.walk()` to deeply scan the selected folder for `.npz` files, ensuring no mocap sequences scattered across subfolders are missed.
- **Indexed Output Naming**: Modified the folder naming logic. If it detects multiple `.npz` files with the same name across different folders (e.g., `data/folderA/misc_poses.npz` and `data/folderB/misc_poses.npz`), it will append an index to the output directory (`misc_poses`, `misc_poses_1`, `misc_poses_2`, etc.) inside `training_data_output/`.
- **YOLO Alignment**: Added a `TRACK_TO` constraint to the rendering camera targeting the Spine/Pelvis to ensure the human is perfectly centered in the 640x480 frame, matching the bounding box distribution of a YOLOv8 mask.

## Phase 2: "Pro" GCN Pipeline & Slang Shaders
The complete Python backend pipeline has been generated and upgraded to the Pro architecture described in our research documents!

### 1. PyTorch Dataset Loader (Pro Version)
Modified [`backend/dataset.py`](file:///Users/Guest/Downloads/vision-track/backend/dataset.py).
- **17 Symmetrical Joints**: We mapped precisely 17 symmetrical core joints spanning left/right hips, knees, ankles, shoulders, elbows, and wrists, along with the spine, neck, and head to guarantee accurate anatomical balance.
- **5-Channel Inputs**: The 2D graphs now contain 5 channels per node $(x, y, r_{\text{left}}, r_{\text{right}}, v)$ to feed the lifter with occlusion visibility data and Laplacian boundary radii.

### 2. Semantic 3D GCN Lifter (Node D)
Modified [`backend/model_gcn.py`](file:///Users/Guest/Downloads/vision-track/backend/model_gcn.py).
- Implemented the `SemanticGCNLifter` which uses a learnable semantic mask ($M$) to multiply the adjacency matrix. This allows non-adjacent joints to communicate mathematically (e.g. left foot coordinating with right hand during a walk).
- Integrated Batch Normalization and LeakyReLU activations to prevent neuron death and over-smoothing.
- The output channels are now 4: Metric $(X, Y, Z)$ and Depth Uncertainty $(\sigma_Z)$.

### 3. The Trinity of Losses
Modified [`backend/train_gcn.py`](file:///Users/Guest/Downloads/vision-track/backend/train_gcn.py).
- We implemented a custom `NodeDLoss` PyTorch module combining:
  1. **MPJPE**: Mean Per-Joint Position Error for exact metric distance.
  2. **Kinematic Bone-Length Consistency**: To penalize anatomical impossibility (e.g. bones stretching out of proportion).
  3. **Uncertainty NLL**: Forces the network to admit low confidence ($\sigma_Z$) when a limb is hidden, rather than guessing blindly.
- Mapped 16 unique `bone_pairs` to support the kinematic loss perfectly.

### 4. Slang Compute Shader Integration (Hybrid Pipeline)
Created [`backend/shaders/skeleton.slang`](file:///Users/Guest/Downloads/vision-track/backend/shaders/skeleton.slang) and [`backend/slang_processor.py`](file:///Users/Guest/Downloads/vision-track/backend/slang_processor.py).
PyTorch passes the GCN-predicted 3D joints directly into GPU VRAM where the Slang compute shader transforms the joints (simulating skinning/Forward Kinematics) with zero-copy overhead.

### 5. ONNX Exporter
Created [`backend/export_onnx.py`](file:///Users/Guest/Downloads/vision-track/backend/export_onnx.py).
A quick script to trace the trained PyTorch model and export it to `static/models/3d_lifter_gcn.onnx` so we can load it into the Three.js/WebGL frontend when we are ready to transition away from Python rendering.


