#!/bin/bash

# Set paths and config
BASE_CONFIG="/home/ssnyde9/dev/VSA-OGM/configs/experiments/exp2_config_toysim_parameters.yaml"
LOG_ROOT="logs"
PROJECT_NAME="VSA-OGM-Parameter-Ablation-ToySim-V2"

# Sweep parameter lists (must match keys in your config)
AXIS_RESOLUTIONS=(1.0)
VECTOR_DIMS=(32768 16384 8192 4096 2048 1024 512 256)
LENGTH_SCALES=(0.1 0.2 0.5 1.0 1.5 2.0)
NUM_TILES=(2 1 4 8)

# Loop over combinations
for AXIS_RES in "${AXIS_RESOLUTIONS[@]}"; do
  for VEC_DIM in "${VECTOR_DIMS[@]}"; do
    for LENGTH_SCALE in "${LENGTH_SCALES[@]}"; do
      for TILE in "${NUM_TILES[@]}"; do

        # Construct experiment name and output directory
        EXP_NAME="axis=${AXIS_RES}_dim=${VEC_DIM}_scale=${LENGTH_SCALE}_tiles=${TILE}"
        EXP_NAME_CLEAN=${EXP_NAME//./-}
        LOG_DIR="${LOG_ROOT}/${PROJECT_NAME}/${EXP_NAME_CLEAN}"

        echo "Launching experiment: $EXP_NAME_CLEAN"

        # Run the evaluation and tee the output
        python drivers/scripts/experiments/run_evaluation_once.py \
          --config $BASE_CONFIG \
          "mapping.axis_resolution=$AXIS_RES" \
          "mapping.vector_dimensionality=$VEC_DIM" \
          "mapping.vector_length_scale=$LENGTH_SCALE" \
          "mapping.num_tiles=$TILE" \
          logging.save_dir=$LOG_DIR \
          experiment_name=$EXP_NAME_CLEAN \
          2>&1 | tee "${LOG_DIR}/log.txt"

      done
    done
  done
done
