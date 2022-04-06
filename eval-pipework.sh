#!/bin/bash

# evaluate 
# pospool_xyz_avg.yaml, pseudo_grid.yaml, 
# pointwisemlp_dp_fi_df_fc1.yaml, adaptiveweight_dp_fc1_avg.yaml
# YAML_FILES=(
	# "pospool_xyz_avg.yaml" 
	# "pospool_sin_cos_avg.yaml" 
	# "pseudo_grid.yaml"
	# "pointwisemlp_dp_fi_df_fc1.yaml"
	# "adaptiveweight_dp_fc1_avg.yaml"
	# "pseudo_grid.yaml"
	# "pseudo_grid.yaml"
	# "pseudo_grid.yaml"
	# "pospool_xyz_avg_se.yaml" 
# )
# CKPT_PATHS=(
	# "pospool_xyz_avg_1628865624/best.pth"
	# "pospool_sin_cos_avg_1629299159/best.pth"
	# "pseudo_grid_1628925913/best.pth"
	# "pointwisemlp_dp_fi_df_fc1_1628956762/best.pth"
	# "adaptiveweight_dp_fc1_avg_1629013720/best.pth"
	# "pseudo_grid_1629643587_new_better/best.pth"
	# "pseudo_grid_1629733517_new_using_CE/best.pth"
	# "pseudo_grid_1629819505_wce/best.pth"
	# "pospool_xyz_avg_se_1629871168/best.pth"
# )
YAML_FILES=(
	"pseudo_grid_se3.yaml"
	"pseudo_grid_se3.yaml"
)
CKPT_PATHS=(
	# "pospool_xyz_avg_1628865624/best.pth"
	"pseudo_grid_se3_1630594159_final_se3_ce"
	"pseudo_grid_se3_1630641193_final_se3_wce"
)

USE_AVG_MAX_POOLS=(
	'true'
	'true'
)

NUM_GPUs=1
DATA_AUGS=("true" "false")
# DATA_AUGS=("true")
# smooth, ce, wce, sqrt_ce
LOSS='smooth'
# false use the default, true use my custom module

for data_aug in "${DATA_AUGS[@]}"; do
	for index in ${!YAML_FILES[*]}; do
		echo "Current yaml: ${YAML_FILES[$index]}"
		echo "Current ckpt: ${CKPT_PATHS[$index]}"

		# we use the GPU 1, CUDA_VISIBLE_DEVICES=1 
		time python -m torch.distributed.launch \
		--master_port 1286 \
		--nproc_per_node ${NUM_GPUs} \
		--local_rank 1 \
		function/evaluate_pipework_dist.py \
		--load_path log/pipework/${CKPT_PATHS[$index]}/best.pth \
		--cfg cfgs/pipework/${YAML_FILES[$index]} \
		--data_aug ${data_aug} \
		--loss ${LOSS} \
		--use_avg_max_pool ${USE_AVG_MAX_POOLS[$index]}  
		# --val_freq 10 

		echo "Current yaml: ${YAML_FILES[$index]}"
		echo "Current ckpt: ${CKPT_PATHS[$index]}"
	done
done