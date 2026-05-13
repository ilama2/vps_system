from pathlib import Path
import random
import shutil
import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to dataset folder"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output folder"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    root = Path(args.root)
    out = Path(args.output)

    train_ratio = 0.80
    test_ratio = 0.20

    seed = 42
    copy_files = True

    rgb_dir = root / "rgb"
    pose_dir = root / "poses"
    calib_dir = root / "calibration"

    rgb_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        rgb_files.extend(rgb_dir.glob(ext))

    rgb_files = sorted(rgb_files)

    samples = []

    for rgb_file in rgb_files:

        stem = rgb_file.stem

        pose_file = pose_dir / f"{stem}.txt"
        calib_file = calib_dir / f"{stem}.txt"

        if not pose_file.exists():
            print(f"Missing pose: {pose_file}")
            continue

        if not calib_file.exists():
            print(f"Missing calibration: {calib_file}")
            continue

        samples.append(
            (rgb_file, pose_file, calib_file)
        )

    print("Valid samples:", len(samples))

    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)

    n_train = int(n * train_ratio)

    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    splits = {
        "train": train_samples,
        "test": test_samples
    }

    for split_name, split_samples in splits.items():

        (out / split_name / "rgb").mkdir(
            parents=True,
            exist_ok=True
        )

        (out / split_name / "poses").mkdir(
            parents=True,
            exist_ok=True
        )

        (out / split_name / "calibration").mkdir(
            parents=True,
            exist_ok=True
        )

        for rgb_file, pose_file, calib_file in split_samples:

            if copy_files:

                shutil.copy2(
                    rgb_file,
                    out / split_name / "rgb" / rgb_file.name
                )

                shutil.copy2(
                    pose_file,
                    out / split_name / "poses" / pose_file.name
                )

                shutil.copy2(
                    calib_file,
                    out / split_name / "calibration" / calib_file.name
                )

            else:

                shutil.move(
                    rgb_file,
                    out / split_name / "rgb" / rgb_file.name
                )

                shutil.move(
                    pose_file,
                    out / split_name / "poses" / pose_file.name
                )

                shutil.move(
                    calib_file,
                    out / split_name / "calibration" / calib_file.name
                )

    print("Done")
    print("Train:", len(train_samples))
    print("Test:", len(test_samples))


if __name__ == "__main__":
    main()