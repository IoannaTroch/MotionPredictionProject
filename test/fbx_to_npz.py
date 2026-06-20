from ai4animation import Motion

# Load from different formats
glb_motion = Motion.LoadFromGLB("character.glb", names=bone_names, floor=None)
fbx_motion = Motion.LoadFromFBX("character.fbx")
bvh_motion = Motion.LoadFromBVH("character.bvh", scale=0.01)

# Load from the internal NPZ format
npz_motion = Motion.LoadFromNPZ("character.npz")

# Save any motion to NPZ
glb_motion.SaveToNPZ("character")