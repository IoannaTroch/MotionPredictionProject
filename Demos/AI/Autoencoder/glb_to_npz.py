from ai4animation import Motion
import os
import sys

# SCRIPT_DIR = Path(__file__).parent
# ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Cranberry")
# sys.path.append(ASSETS_PATH)


# Φόρτωση από GLB (παρέχοντας τα ονόματα των οστών/bones)
glb_motion = Motion.LoadFromGLB("Sh20_cranberry_s01.running_S01.trk.glb")

# Αποθήκευση της κίνησης σε μορφή NPZ
glb_motion.SaveToNPZ("Sh20_cranberry_s01.running_S01.trk")