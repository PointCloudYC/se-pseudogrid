#!/bin/bash

# train
# NUM_POINTS=(15000 8192 4096)
NUM_POINTS=10000
NUM_GPUs=2

# pospool_xyz_avg.yaml, pseudo_grid.yaml, 
# pointwisemlp_dp_fi_df_fc1.yaml, adaptiveweight_dp_fc1_avg.yaml
# YAML_FILES=("pospool_xyz_avg.yaml" "pseudo_grid.yaml" "pointwisemlp_dp_fi_df_fc1.yaml" "adaptiveweight_dp_fc1_avg.yaml")
# new SE-CLoserLook3D: pospool_xyz_avg_se, 
YAML_FILES=("pseudo_grid_se3.yaml")
use_avg_max_pool='true'

# ablation study on SE choices including SE_squeeze_type and SE_excitation_type
SQUEEZE_TYPES=('max') # default is avg
for squeeze_type in "${SQUEEZE_TYPES[@]}"; do
    for yaml_file in "${YAML_FILES[@]}"; do

        echo "squeeze type: ${squeeze_type}"
        echo "yaml file: ${yaml_file}"

        time python -m torch.distributed.launch \
        --master_port 12345678 \
        --nproc_per_node ${NUM_GPUs} \
        function/train_pipework_dist.py \
        --cfg cfgs/pipework/${yaml_file} \
        --num_points ${NUM_POINTS} \
        --val_freq 10 \
        --save_freq 50 \
        --use_avg_max_pool ${use_avg_max_pool}  \
        --SE_squeeze_type ${squeeze_type}
        # --load_path log/pipework/pseudo_grid_1629560283/ckpt_epoch_320.pth \
        # --start_epoch 320
        # [--log_dir <dir>] \
        # --batch_size ${batch_size} \

        echo "squeeze type: ${squeeze_type}"
        echo "yaml file: ${yaml_file}"
    done
done

EXCITATION_TYPES=('relu' 'tanh') # default is sigmoid
for exictiation_type in "${EXCITATION_TYPES[@]}"; do
    for yaml_file in "${YAML_FILES[@]}"; do

        echo "excitation type: ${exictiation_type}"
        echo "yaml file: ${yaml_file}"

        time python -m torch.distributed.launch \
        --master_port 12345678 \
        --nproc_per_node ${NUM_GPUs} \
        function/train_pipework_dist.py \
        --cfg cfgs/pipework/${yaml_file} \
        --num_points ${NUM_POINTS} \
        --val_freq 10 \
        --save_freq 50 \
        --use_avg_max_pool ${use_avg_max_pool}  \
        --SE_excitation_type ${exictiation_type}
        # --load_path log/pipework/pseudo_grid_1629560283/ckpt_epoch_320.pth \
        # --start_epoch 320
        # [--log_dir <dir>] \
        # --batch_size ${batch_size} \

        echo "excitation type: ${exictiation_type}"
        echo "yaml file: ${yaml_file}"
    done
done