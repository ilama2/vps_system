from pathlib import Path
import argparse
import shutil
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert COLMAP TXT sparse model to ACE-G dataset format")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--sparse_txt_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def qvec_to_rotmat(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])


def read_cameras(cameras_txt):
    cameras = {}

    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            p = line.split()
            cam_id = int(p[0])
            model = p[1]
            params = list(map(float, p[4:]))

            if model == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx, fy = f, f
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_RADIAL":
                f, cx, cy, _ = params[:4]
                fx, fy = f, f
            elif model == "OPENCV":
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            cameras[cam_id] = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])

    return cameras


def read_images(images_txt):
    images = []
    lines = open(images_txt, "r").readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "" or line.startswith("#"):
            i += 1
            continue

        p = line.split()

        images.append({
            "qw": float(p[1]),
            "qx": float(p[2]),
            "qy": float(p[3]),
            "qz": float(p[4]),
            "tx": float(p[5]),
            "ty": float(p[6]),
            "tz": float(p[7]),
            "camera_id": int(p[8]),
            "name": p[9],
        })

        i += 2

    return images


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    sparse_txt_dir = Path(args.sparse_txt_dir)
    output = Path(args.output)

    out_rgb = output / "rgb"
    out_poses = output / "poses"
    out_calib = output / "calibration"

    out_rgb.mkdir(parents=True, exist_ok=True)
    out_poses.mkdir(parents=True, exist_ok=True)
    out_calib.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras(sparse_txt_dir / "cameras.txt")
    images = read_images(sparse_txt_dir / "images.txt")

    for img in images:
        name = img["name"]
        stem = Path(name).stem

        src_img = image_dir / name

        if not src_img.exists():
            print("Missing image:", src_img)
            continue

        shutil.copy2(src_img, out_rgb / name)

        R = qvec_to_rotmat(
            img["qw"],
            img["qx"],
            img["qy"],
            img["qz"]
        )

        t = np.array(
            [img["tx"], img["ty"], img["tz"]],
            dtype=np.float64
        ).reshape(3, 1)

        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = R
        pose_w2c[:3, 3:4] = t

        pose_c2w = np.linalg.inv(pose_w2c)

        np.savetxt(out_poses / f"{stem}.txt", pose_c2w, fmt="%.10f")
        np.savetxt(out_calib / f"{stem}.txt", cameras[img["camera_id"]], fmt="%.10f")

    print("Done.")
    print("RGB:", len(list(out_rgb.glob("*"))))
    print("Poses:", len(list(out_poses.glob("*.txt"))))
    print("Calibration:", len(list(out_calib.glob("*.txt"))))


if __name__ == "__main__":
    main()