#!/bin/bash
set -e


ACEG_ROOT="/workspace/ace-g"
DATASET_ROOT="/workspace/dataset"
CONFIG="/workspace/ace-g/src/ace_g/configs/ace_g_25min.yaml"
OUTPUT_DIR="/workspace/ace-g/my_outputs"

TRAIN_RGB="$DATASET_ROOT/train/rgb/*.jpg"
TRAIN_POSES="$DATASET_ROOT/train/poses/*.txt"
TRAIN_CALIB="$DATASET_ROOT/train/calibration/*.txt"

TEST_RGB="$DATASET_ROOT/test/rgb/*.jpg"
TEST_POSES="$DATASET_ROOT/test/poses/*.txt"
TEST_CALIB="$DATASET_ROOT/test/calibration/*.txt"

export PYTHONPATH="$ACEG_ROOT/src"


#  Train ACE-G 
python3 -m ace_g.train_single_scene \
  --config "$CONFIG" \
  --dataset.rgb_files "$TRAIN_RGB" \
  --dataset.pose_files "$TRAIN_POSES" \
  --dataset.calibration_files "$TRAIN_CALIB" \
  --output_dir "$OUTPUT_DIR"
  --use_rerun True \
  --rerun_spawn False


#   test images
python3 -m ace_g.register_images \
  --config "$MAP_CONFIG" \
  --dataset.rgb_files "$TEST_RGB" \
  --dataset.calibration_files "$TEST_CALIB"



#  Evaluate poses
python3 -m ace_g.eval_poses \
  --estimated_pose_file "$EST_POSES" \
  --gt_pose_files "$TEST_POSES"

