#!/bin/bash
# Fix 2: Hidden-state VCD — layer L에서 two-branch hidden state 결합
# h_star = (1+α)·h_normal_L − α·h_noisy_L → 남은 레이어 통과 → logit
#
# 사용법:
#   bash cd_scripts/llava1.5_pope_hidden_vcd.bash [seed] [dataset] [type] [model_path] [proj_layer] [cd_alpha]

seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"popular"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
proj_layer=${5:-14}
cd_alpha=${6:-1.0}

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/sjkim/datasets/coco/val2014
else
  image_folder=/home/sjkim/datasets/gqa/images
fi

/home/sjkim/miniconda3/envs/vl/bin/python ./eval/object_hallucination_vqa_llava_hidden_vcd.py \
  --model-path ${model_path} \
  --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
  --image-folder ${image_folder} \
  --answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_hidden_vcd_L${proj_layer}_a${cd_alpha}_seed${seed}.jsonl \
  --noise_step 500 \
  --cd_alpha ${cd_alpha} \
  --cd_beta 0.1 \
  --proj_layer ${proj_layer} \
  --seed ${seed}
