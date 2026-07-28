import csv
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation

FOOT_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

traj_fp = os.path.join(os.path.dirname(__file__), "traj_hopscotch_friction_6cm_lsq.json")
output_fp = os.path.join(os.path.dirname(__file__), "traj_hopscotch_friction_6cm_lsq.csv")

with open(traj_fp, 'r') as file:
    traj = json.load(file)

with open(output_fp, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    for mode in traj:
        contact_row = [1.0 if foot in mode["contacts"] else 0.0 for foot in FOOT_ORDER]
        for q in mode['q']:
            q_go2 = np.zeros(23)  # was 19; +4 for contacts
            q_go2[0:3] = q[0:3]
            q_go2[3:7] = Rotation.from_euler("XYZ", q[3:6]).as_quat()
            q_go2[7:19] = q[6:18]
            q_go2[19:23] = contact_row
            writer.writerow(q_go2)