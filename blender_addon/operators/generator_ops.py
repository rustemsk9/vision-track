import bpy
import math
import json
import os
import random
import logging
from bpy_extras.io_utils import ImportHelper

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
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Starting Batch Generation from directory: {directory}")

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
            if obj.type == 'MESH' and (obj.parent == smpl_rig or "SMPL" in obj.name or "Mesh" in obj.name or obj.name.startswith("SMPL")):
                smpl_mesh = obj
                break
                
        if not smpl_mesh:
            self.report({'WARNING'}, "Could not definitively find the SMPL Mesh. Zero-Raytracing Masking will be skipped.")
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
            
            unique_name = base_name
            index = 1
            while unique_name in used_names or os.path.exists(os.path.join(output_dir, unique_name)):
                unique_name = f"{base_name}_{index}"
                index += 1
            
            used_names.add(unique_name)
            seq_output_dir = os.path.join(output_dir, unique_name)
            
            logger.info(f"Processing sequence: {unique_name} (from {filepath})")
            
            cam_obj = bpy.data.objects["iMacCamera"]
            cam_obj.location = (0, -3.5, 1.2)
            cam_obj.rotation_euler = (math.radians(90), 0, 0)
            
            smpl_rig.rotation_mode = 'XYZ'
            smpl_rig.rotation_euler = (math.radians(90), math.radians(180), 0)
            
            logger.info("Camera and Rig coordinates locked for standard Z-up front-facing posture")
            
            num_frames = self.load_amass_data(smpl_rig, filepath)
            
            # Export animation sequence (frame_step=10 to subsample MoCap frames)
            frames_to_render = num_frames
            self.export_training_data(seq_output_dir, 1, frames_to_render, smpl_rig, frame_step=10)
            logger.info(f"Finished exporting {frames_to_render} frames for {base_name}")
            
        self.report({'INFO'}, f"Batch Generation Complete! Check {output_dir}")
        logger.info("Batch Generation Complete!")
        
        logging.shutdown()
        return {'FINISHED'}

    def setup_imac_camera(self, target_rig=None):
        if "iMacCamera" in bpy.data.objects:
            cam_obj = bpy.data.objects["iMacCamera"]
        else:
            cam_data = bpy.data.cameras.new("iMacCamera")
            cam_obj = bpy.data.objects.new("iMacCamera", cam_data)
            bpy.context.collection.objects.link(cam_obj)
        
        cam_obj.data.lens = 28 
        cam_obj.location = (0, -2.5, 1.2)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)
        bpy.context.scene.camera = cam_obj
        
        if target_rig:
            cam_obj.constraints.clear()
            track_const = cam_obj.constraints.new(type='TRACK_TO')
            track_const.target = target_rig
            
            if 'Spine1' in target_rig.pose.bones:
                track_const.subtarget = 'Spine1'
            elif 'Pelvis' in target_rig.pose.bones:
                track_const.subtarget = 'Pelvis'
                
            track_const.track_axis = 'TRACK_NEGATIVE_Z'
            track_const.up_axis = 'UP_Y'

    def load_amass_data(self, rig, filepath):
        """Loads AMASS .npz file and keyframes all pose bone rotations (Upper & Lower Body)"""
        import numpy as np
        import mathutils
        
        data = np.load(filepath)
        poses = data['poses']
        trans = data['trans']
        
        num_frames = poses.shape[0]
        
        # SMPL 24 Joint Hierarchy
        joint_names = [
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
            'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 
            'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
            'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
        ]
        
        if rig.animation_data:
            rig.animation_data_clear()
            
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = num_frames
        
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='POSE')
        
        for f in range(num_frames):
            frame_idx = f + 1
            
            pelvis_bone = rig.pose.bones.get('Pelvis')
            if pelvis_bone:
                pelvis_bone.location = (trans[f][0], trans[f][1], trans[f][2])
                pelvis_bone.keyframe_insert(data_path="location", frame=frame_idx)
            
            for i, joint_name in enumerate(joint_names):
                bone = rig.pose.bones.get(joint_name)
                if not bone:
                    continue
                    
                start_idx = i * 3
                axis_angle = poses[f, start_idx:start_idx+3]
                
                angle = np.linalg.norm(axis_angle)
                if angle > 1e-6:
                    axis = axis_angle / angle
                    quat = mathutils.Quaternion(axis, angle)
                else:
                    quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
                
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = quat
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
                
        bpy.ops.object.mode_set(mode='OBJECT')
        return num_frames

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

    def export_training_data(self, output_dir, frame_start, frame_end, smpl_obj, frame_step=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        joint_data = []
        
        for f in range(frame_start, frame_end + 1, frame_step):
            bpy.context.scene.frame_set(f)
            filepath = os.path.join(output_dir, f"mask_{f:04d}.png")
            bpy.context.scene.render.filepath = filepath
            
            bpy.ops.render.render(write_still=True)
            
            frame_joints = {}
            if smpl_obj.type == 'ARMATURE':
                for bone in smpl_obj.pose.bones:
                    global_pos = smpl_obj.matrix_world @ bone.head
                    frame_joints[bone.name] = [global_pos.x, global_pos.y, global_pos.z]
                    
            joint_data.append({"frame": f, "joints": frame_joints})
            
        joints_path = os.path.join(output_dir, "joints.jsonl")
        with open(joints_path, "w") as f:
            for entry in joint_data:
                f.write(json.dumps(entry) + "\n")

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
