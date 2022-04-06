#!/bin/bash

YAML_FILES=(
	"pseudo_grid_se3.yaml"
	"pseudo_grid_se3.yaml"
	"pseudo_grid_se3.yaml"
)
CKPT_PATHS=(
	"pseudo_grid_se3_1633789944_SE_squeeze_max"
	"pseudo_grid_se3_1633848952_SE_exicitation_relu"
	"pseudo_grid_se3_1633906978"
)

USE_AVG_MAX_POOLS=(
	'true'
	'true'
	'true'
)

NUM_GPUs=1
DATA_AUGS=("true" "false")

# SE ablation study
SE_squeeze_types=(
	'max'
	'avg'
	'avg'
)
SE_excitation_types=(
	'sigmoid'
	'relu'
	'tanh'
)

# smooth, ce, wce, sqrt_ce
LOSS='smooth'

for data_aug in "${DATA_AUGS[@]}"; do
	for index in ${!YAML_FILES[*]}; do
		echo "Current yaml: ${YAML_FILES[$index]}"
		echo "Current ckpt: ${CKPT_PATHS[$index]}"

		time python -m torch.distributed.launch \
		--master_port 1234 \
		--nproc_per_node ${NUM_GPUs} \
		function/evaluate_pipework_dist.py \
		--load_path log/pipework/${CKPT_PATHS[$index]}/best.pth \
		--cfg cfgs/pipework/${YAML_FILES[$index]} \
		--data_aug ${data_aug} \
		--loss ${LOSS} \
		--use_avg_max_pool ${USE_AVG_MAX_POOLS[$index]}  \
		--SE_squeeze_type ${SE_squeeze_types[$index]} \
		--SE_excitation_type ${SE_excitation_types[$index]}
		# --val_freq 10 

		echo "Current yaml: ${YAML_FILES[$index]}"
		echo "Current ckpt: ${CKPT_PATHS[$index]}"
	done
done