bl_info = {
    "name": "VisionTrack Generator",
    "author": "VisionTrack Team",
    "description": "Generate synthetic 3D pose data from AMASS BVH with Zero-Raytracing Masking",
    "blender": (4, 0, 0),
    "version": (0, 1, 0),
    "location": "View3D > Sidebar > VisionTrack",
    "category": "Development"
}

from . import auto_load

auto_load.init()

def register():
    auto_load.register()

def unregister():
    auto_load.unregister()
