import os
import numpy as np
import dataset_utils as dutil

# -------------------------
# CONFIG
# -------------------------
RAW_DIR = "7scenes_raw"
OUT_DIR = "7scenes"
SCENE = "chess"

FOCAL_LENGTH = 525.0

# depth → RGB calibration matrix
d_to_rgb = np.array([
    [0.999965,  0.00267, -0.00790, -0.0255],
    [-0.00274,  0.999963, -0.00815,  0.0001],
    [0.00788,   0.00817,  0.999935,  0.0020],
    [0, 0, 0, 1]
])

# -------------------------
# PROCESS SPLIT
# -------------------------
def process_split(split_name):

    scene_path = f"{RAW_DIR}/{SCENE}"
    target = f"{OUT_DIR}/kf_chess/{split_name}"

    os.makedirs(f"{target}/rgb", exist_ok=True)
    os.makedirs(f"{target}/poses", exist_ok=True)
    os.makedirs(f"{target}/calibration", exist_ok=True)

    # read train/test split
    with open(f"{scene_path}/{split_name.capitalize()}Split.txt", "r") as f:
        seqs = ["seq-" + s.strip()[8:].zfill(2) for s in f.readlines()]

    for seq in seqs:

        seq_path = f"{scene_path}/{seq}"
        files = os.listdir(seq_path)

        # -------------------------
        # RGB IMAGES
        # -------------------------
        rgb_files = [f for f in files if f.endswith("color.png")]

        for img in rgb_files:
            os.system(
                f"ln -s ../../{seq_path}/{img} "
                f"{target}/rgb/{seq}-{img}"
            )

        # -------------------------
        # POSES (convert to RGB frame)
        # -------------------------
        pose_files = [f for f in files if f.endswith("pose.txt")]

        for p in pose_files:

            pose = np.loadtxt(f"{seq_path}/{p}")

            # depth → RGB transform
            pose = pose @ np.linalg.inv(d_to_rgb)

            dutil.write_cam_pose(
                f"{target}/poses/{seq}-{p}",
                pose
            )

        # -------------------------
        # CALIBRATION
        # -------------------------
        for i in range(len(rgb_files)):

            dutil.write_focal_length(
                f"{target}/calibration/{seq}-frame-{str(i).zfill(6)}.txt",
                FOCAL_LENGTH
            )

# -------------------------
# RUN
# -------------------------
process_split("train")
process_split("test")

print("DONE: chess dataset prepared.")