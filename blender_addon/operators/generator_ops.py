import bpy
import math
import json
import os
import random
import logging
from bpy_extras.io_utils import ImportHelper
from bpy_extras.object_utils import world_to_camera_view

# The 17 GCN joints in exact inference order — must match dataset.py and JS gcnNodes[]
GCN_JOINT_NAMES = [
    'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle',
    'L_Hip', 'L_Knee', 'L_Ankle',
    'Spine1', 'Spine2', 'Neck', 'Head',
    'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Shoulder', 'R_Elbow', 'R_Wrist'
]

class VISIONTRACK_OT_generate_data(bpy.types.Operator, ImportHelper):
    """Generate VisionTrack Synthetic Data (Batch Mode)"""
    bl_idname = "visiontrack.generate_data"
    bl_label = "Batch Generate Synthetic Data"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(
        name="AMASS Folder",
        description="Path to the folder containing AMASS .npz files",
        default="",
        maxlen=1024,
        subtype='DIR_PATH'
    )
    
    filter_folder: bpy.props.BoolProperty(
        default=True,
        options={'HIDDEN'}
    )

    def execute(self, context):
        directory = self.filepath
        if not os.path.isdir(directory):
            directory = os.path.dirname(directory)

        if not directory or not os.path.exists(directory):
            self.report({'ERROR'}, "Invalid AMASS directory selected.")
            return {'CANCELLED'}

        output_dir = bpy.path.abspath("//training_data_output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        log_file = os.path.join(output_dir, "visiontrack_generator.log")
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', force=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Starting Batch Generation from directory: {directory}")

        # Dynamic Rig Detection
        smpl_rig = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE' and 'Pelvis' in obj.pose.bones:
                smpl_rig = obj
                break

        if not smpl_rig:
            msg = "Could not find any Armature with a 'Pelvis' bone! Please import the SMPL base model first."
            self.report({'ERROR'}, msg)
            logger.error(msg)
            return {'CANCELLED'}

        logger.info(f"Found SMPL Rig: {smpl_rig.name}")
        self.setup_imac_camera(smpl_rig)

        smpl_mesh = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH' and (obj.parent == smpl_rig or "SMPL" in obj.name
                                        or "Mesh" in obj.name or obj.name.startswith("SMPL")):
                smpl_mesh = obj
                break

        if not smpl_mesh:
            self.report({'WARNING'}, "Could not find the SMPL Mesh. Zero-Raytracing Masking will be skipped.")
            logger.warning("SMPL Mesh not found.")
        else:
            logger.info(f"Found SMPL Mesh: {smpl_mesh.name}")
            self.apply_zero_raytracing_masking(smpl_mesh)

        npz_files = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith('.npz'):
                    npz_files.append(os.path.join(root, f))

        if not npz_files:
            msg = f"No .npz files found in {directory}"
            self.report({'ERROR'}, msg)
            logger.error(msg)
            return {'CANCELLED'}

        logger.info(f"Found {len(npz_files)} .npz files for batch processing.")

        # Render settings
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        bpy.context.scene.view_settings.view_transform = 'Raw'
        bpy.context.scene.display.shading.light = 'FLAT'
        bpy.context.scene.display.shading.color_type = 'SINGLE'
        bpy.context.scene.display.shading.single_color = (1.0, 1.0, 1.0)
        bpy.context.scene.display.shading.background_type = 'VIEWPORT'
        bpy.context.scene.display.shading.background_color = (0.0, 0.0, 0.0)
        bpy.context.scene.render.image_settings.color_mode = 'BW'
        bpy.context.scene.render.image_settings.compression = 100
        bpy.context.scene.render.resolution_x = 640
        bpy.context.scene.render.resolution_y = 480
        bpy.context.scene.render.resolution_percentage = 100

        used_names = set()
        for filepath in npz_files:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            # Include parent folder (subject ID e.g. rub094) to guarantee unique names
            subject_id = os.path.basename(os.path.dirname(filepath))
            combined_name = f"{subject_id}_{base_name}"

            unique_name = combined_name
            index = 1
            while unique_name in used_names or os.path.exists(os.path.join(output_dir, unique_name)):
                unique_name = f"{combined_name}_{index}"
                index += 1

            used_names.add(unique_name)
            seq_output_dir = os.path.join(output_dir, unique_name)

            logger.info(f"Processing sequence: {unique_name} (from {filepath})")

            smpl_rig.rotation_mode = 'XYZ'
            # (90X) = stand upright, (180Y) = face forward, (90Z) = face camera
            smpl_rig.rotation_euler = (math.radians(90), math.radians(180), math.radians(90))

            # Single-pass: apply pose directly each frame, render, extract data.
            # NO keyframe_insert() calls — eliminates the 15k+ frame NLA stall.
            self.process_amass_sequence(smpl_rig, filepath, seq_output_dir, frame_step=10)
            logger.info(f"Finished sequence: {unique_name}")

        self.report({'INFO'}, f"Batch Generation Complete! Check {output_dir}")
        logger.info("Batch Generation Complete!")
        logging.shutdown()
        return {'FINISHED'}

    def setup_imac_camera(self, target_rig=None):
        """
        Sets up a FIXED static iMac-style camera.
        NO Track To constraint — the camera never moves or rotates.
        The character is kept centered in frame by zeroing horizontal translation
        in load_amass_data(), so tracking is unnecessary and harmful for training.
        """
        if "iMacCamera" in bpy.data.objects:
            cam_obj = bpy.data.objects["iMacCamera"]
        else:
            cam_data = bpy.data.cameras.new("iMacCamera")
            cam_obj = bpy.data.objects.new("iMacCamera", cam_data)
            bpy.context.collection.objects.link(cam_obj)

        cam_obj.data.lens = 28
        # Fixed position: 3.5m in front of origin, at chest height (1.2m)
        cam_obj.location = (0, -3.5, 1.2)
        # Pointing straight ahead along +Y axis (90° on X = standard Blender front view)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)
        bpy.context.scene.camera = cam_obj

        # Remove ALL constraints — no Track To, no Follow Path, nothing.
        # A moving camera produces inconsistent projection angles across frames,
        # which would corrupt the training data.
        cam_obj.constraints.clear()

    def process_amass_sequence(self, rig, filepath, output_dir, frame_step=10):
        """
        Single-pass AMASS processor: applies pose directly to bones each frame,
        renders the mask, and extracts 2D/3D data — all in one loop.

        NEVER calls keyframe_insert(). The old two-pass approach (load_amass_data
        keyframing every frame, then export_training_data rendering every 10th)
        caused Blender to stall on long sequences (15k+ keyframe insertions in
        the NLA system). This single-pass approach is 10-100x faster.
        """
        import numpy as np
        import mathutils

        data = np.load(filepath)
        poses = data['poses']
        trans = data['trans']
        num_frames = poses.shape[0]

        joint_names = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
            'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar',
            'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
            'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        ]

        # Clear any existing animation so it doesn't interfere with direct bone setting
        if rig.animation_data:
            rig.animation_data_clear()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        scene = bpy.context.scene
        cam_obj = scene.camera
        joint_data = []

        # Switch to Pose mode once before the loop
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='POSE')

        for f in range(0, num_frames, frame_step):
            # --- Directly apply AMASS pose to bones (NO keyframe_insert) ---
            pelvis_bone = rig.pose.bones.get('Pelvis')
            if pelvis_bone:
                # Treadmill Effect: Lock X and Y to 0.0 so subject stays centered; keep Z (Height) for squats/jumps
                pelvis_bone.location = (0.0, 0.0, float(trans[f][2]))

            for i, joint_name in enumerate(joint_names):
                bone = rig.pose.bones.get(joint_name)
                if not bone:
                    continue
                axis_angle = poses[f, i * 3:(i + 1) * 3]
                angle = float(np.linalg.norm(axis_angle))
                if angle > 1e-6:
                    quat = mathutils.Quaternion(
                        (float(axis_angle[0] / angle),
                         float(axis_angle[1] / angle),
                         float(axis_angle[2] / angle)),
                        angle
                    )
                else:
                    quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = quat

            # Exit pose mode so the render picks up the updated pose
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.update()

            # --- Render mask ---
            frame_label = f + 1  # 1-indexed label for filename
            mask_path = os.path.join(output_dir, f"mask_{frame_label:04d}.png")
            scene.render.filepath = mask_path
            bpy.ops.render.render(write_still=True)

            # --- Get evaluated objects (constraints + pose fully resolved) ---
            depsgraph = bpy.context.evaluated_depsgraph_get()
            rig_eval = rig.evaluated_get(depsgraph)
            cam_eval = cam_obj.evaluated_get(depsgraph) if cam_obj else None

            # --- 3D joint positions IN CAMERA VIEW SPACE ---
            joints_3d = {}
            if cam_eval is not None:
                inv_cam_matrix = cam_eval.matrix_world.inverted()
                for bone in rig_eval.pose.bones:
                    world_pos = rig_eval.matrix_world @ bone.head
                    cam_pos = inv_cam_matrix @ world_pos
                    # Map Blender Camera Space -> WebGL standard [X_right, Y_up(height), Z_depth]
                    joints_3d[bone.name] = [
                        float(cam_pos.x),
                        float(cam_pos.z),
                        float(-cam_pos.y)
                    ]
            else:
                for bone in rig_eval.pose.bones:
                    gp = rig_eval.matrix_world @ bone.head
                    joints_3d[bone.name] = [float(gp.x), float(gp.y), float(gp.z)]

            # --- 2D projected keypoints (via evaluated camera, no Track To jitter) ---
            keypoints_2d = {}
            if cam_eval is not None:
                for bone_name in GCN_JOINT_NAMES:
                    keypoints_2d[bone_name] = self.project_bone_to_2d(
                        scene, cam_eval, rig_eval, bone_name
                    )

            joint_data.append({
                "frame": frame_label,
                "joints": joints_3d,
                "joints_3d": joints_3d,
                "keypoints_2d": keypoints_2d
            })

            # Return to Pose mode for next iteration
            bpy.context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode='POSE')

        # Exit pose mode after all frames processed
        bpy.ops.object.mode_set(mode='OBJECT')

        # Write joints.jsonl
        joints_path = os.path.join(output_dir, "joints.jsonl")
        with open(joints_path, "w") as jf:
            for entry in joint_data:
                jf.write(json.dumps(entry) + "\n")

        return len(joint_data)

    def project_bone_to_2d(self, scene, cam_eval, rig_eval, bone_name):
        """
        Projects a 3D SMPL bone head position through the Blender camera
        to get 2D pixel coordinates (u, v) in [0, render_width] x [0, render_height].

        IMPORTANT: cam_eval and rig_eval must be EVALUATED objects from the depsgraph
        (not the base objects). This ensures the camera's Track To constraint rotation
        is fully applied at the current frame before projection — otherwise the
        camera matrix is stale and projections will be wrong.

        This is the training-time equivalent of what YOLOv8-pose outputs at inference time.
        """
        bone = rig_eval.pose.bones.get(bone_name)
        if bone is None:
            # Return image center as fallback
            return [320.0, 240.0]

        # Get the bone head position in world space using the EVALUATED rig matrix
        # (reflects current-frame pose after all constraints and drivers)
        world_pos = rig_eval.matrix_world @ bone.head

        # Project through the EVALUATED camera matrix
        # (reflects the Track To constraint rotation at this exact frame)
        cam_coord = world_to_camera_view(scene, cam_eval, world_pos)

        # Convert normalized camera coords [0,1] to pixel coords
        render = scene.render
        render_w = render.resolution_x * render.resolution_percentage / 100.0
        render_h = render.resolution_y * render.resolution_percentage / 100.0

        # cam_coord.x = [0,1] left to right, cam_coord.y = [0,1] bottom to top
        u = cam_coord.x * render_w
        v = (1.0 - cam_coord.y) * render_h  # Flip Y: Blender camera Y is bottom-up

        return [u, v]

    def apply_zero_raytracing_masking(self, obj):
        if "ZeroRaytracingMasking" not in obj.modifiers:
            modifier = obj.modifiers.new(name="ZeroRaytracingMasking", type='NODES')
            node_group = bpy.data.node_groups.new('ZeroRaytracingNodes', 'GeometryNodeTree')
            modifier.node_group = node_group

            node_group.nodes.clear()
            node_input = node_group.nodes.new('NodeGroupInput')
            node_output = node_group.nodes.new('NodeGroupOutput')
            if hasattr(node_group, "interface"):
                node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
                node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
            else:
                node_group.outputs.new('NodeSocketGeometry', "Geometry")
                node_group.inputs.new('NodeSocketGeometry', "Geometry")

            node_group.links.new(node_input.outputs[0], node_output.inputs[0])




class VISIONTRACK_PT_panel(bpy.types.Panel):
    """Creates a Panel in the 3D Viewport"""
    bl_label = "VisionTrack Generator"
    bl_idname = "VISIONTRACK_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VisionTrack'

    def draw(self, context):
        layout = self.layout
        layout.label(text="Select AMASS Folder:")
        layout.operator("visiontrack.generate_data", icon='PLAY', text="Batch Generate Data")
