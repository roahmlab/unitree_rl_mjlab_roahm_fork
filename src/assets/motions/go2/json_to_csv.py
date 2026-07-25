import csv
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation

# Trajectory path
traj_fp = os.path.join(os.path.dirname(__file__), "traj_hopscotch_friction_6cm_lsq.json")
output_fp = os.path.join(os.path.dirname(__file__), "traj_hopscotch_friction_6cm_lsq.csv")

# Load trajectory
with open(traj_fp, 'r') as file:
    traj = json.load(file)

# Create and write to the CSV file
with open(output_fp, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    for mode in traj:
        for q in mode['q']:
            q_go2 = np.zeros(19)
            q_go2[0:3] = q[0:3]
            q_go2[3:7] = Rotation.from_euler("XYZ", q[3:6]).as_quat()
            q_go2[7:19] = q[6:18]
            writer.writerow(q_go2)