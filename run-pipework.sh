#!/bin/bash

# train
# NUM_POINTS=(15000 8192 4096)
NUM_POINTS=(10000)
# BATCH_SIZES=(16 32)
BATCH_SIZES=("Unknown")
NUM_GPUs=2
# pospool_xyz_avg.yaml, pseudo_grid.yaml, 
# pointwisemlp_dp_fi_df_fc1.yaml, adaptiveweight_dp_fc1_avg.yaml
# YAML_FILES=("pospool_xyz_avg.yaml" "pseudo_grid.yaml" "pointwisemlp_dp_fi_df_fc1.yaml" "adaptiveweight_dp_fc1_avg.yaml")
# new SE-CLoserLook3D: pospool_xyz_avg_se, 
YAML_FILES=("pospool_xyz_avg_se3.yaml")
# ce, smooth, wce, sqrt_ce
# LOSSES=('smooth' 'wce' 'sqrt_ce')
# LOSSES=('ce' 'sqrt_ce')
LOSSES=('ce')
# false use the default, true use my custom module
use_avg_max_pool='true'

for num_points in "${NUM_POINTS[@]}"; do
    for yaml_file in "${YAML_FILES[@]}"; do
        for loss in "${LOSSES[@]}"; do

            # echo "batch_size: ${batch_size}"
            echo "num_points: ${num_points}"
            echo "yaml file: ${yaml_file}"
            echo "loss type: ${loss}"

            time python -m torch.distributed.launch \
            --master_port 12345678 \
            --nproc_per_node ${NUM_GPUs} \
            function/train_pipework_dist.py \
            --cfg cfgs/pipework/${yaml_file} \
            --num_points ${num_points} \
            --val_freq 10 \
            --save_freq 50 \
            --loss ${loss} \
            --use_avg_max_pool ${use_avg_max_pool}  
            # --load_path log/pipework/pseudo_grid_1629560283/ckpt_epoch_320.pth \
            # --start_epoch 320
            # [--log_dir <dir>] \
            # --batch_size ${batch_size} \

            # echo "batch_size: ${batch_size}"
            echo "num_points: ${num_points}"
            echo "yaml file: ${yaml_file}"
            echo "loss type: ${loss}"
        done
    done
done
