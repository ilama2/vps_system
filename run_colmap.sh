#!/bin/bash

export QT_QPA_PLATFORM=offscreen

colmap feature_extractor \
  --database_path database.db \
  --image_path images \
  --ImageReader.single_camera 0 \
  --SiftExtraction.use_gpu 0
  --SiftExtraction.max_num_features 8000 \

colmap sequential_matcher \
  --database_path database.db \
  --SequentialMatching.overlap 10 \
  --SiftMatching.use_gpu 0

colmap mapper \
  --database_path database.db \
  --image_path images \
  --output_path sparse \
  --Mapper.multiple_models 0

colmap model_converter \
  --input_path sparse/0 \
  --output_path sparse_txt \
  --output_type TXT

