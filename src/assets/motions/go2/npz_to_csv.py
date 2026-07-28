import os
import numpy as np
from scipy.spatial.transform import Rotation

# Must match the column order csv_to_npz.py expects (from its explicit
# joint_names list) and the FOOT_ORDER used elsewhere in the pipeline.
JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
FOOT_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

npz_fp = os.path.join(os.path.dirname(__file__), "sys_id_hopscotch.npz")
output_fp = os.path.join(os.path.dirname(__file__), "sys_id_hopscotch.csv")

data = np.load(npz_fp)
q = data["q"]                        # [T, 18]
contact = data["contact"]            # [T, 4]
joint_names = list(data["joint_names"])
feet = list(data["feet"])

# Resolve columns by name rather than trusting position, in case this npz's
# column order ever drifts from JOINT_ORDER/FOOT_ORDER.
joint_col_idx = [joint_names.index(name) for name in JOINT_ORDER]
foot_col_idx = [feet.index(name) for name in FOOT_ORDER]

T = q.shape[0]
out = np.zeros((T, 23), dtype=np.float64)
out[:, 0:3] = q[:, 0:3]                                        # base position
out[:, 3:7] = Rotation.from_euler("XYZ", q[:, 3:6]).as_quat()   # base Euler -> quat xyzw
out[:, 7:19] = q[:, joint_col_idx]                              # 12 joint positions
out[:, 19:23] = contact[:, foot_col_idx].astype(np.float64)     # 4 foot contact flags

np.savetxt(output_fp, out, delimiter=",")

rate = float(data["rate"])
print(f"Wrote {T} frames ({T / rate:.3f}s @ {rate:.0f} Hz) to {output_fp}")