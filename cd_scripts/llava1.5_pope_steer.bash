#!/bin/bash
# Fix 1: Additive Steering — h -= γ·d̂_abl (고정 상수 이동, 투영 아님)
#
# 사용법:
#   bash cd_scripts/llava1.5_pope_steer.bash [seed] [dataset] [type] [model_path] [proj_layer] [proj_gamma]

seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"popular"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
proj_layer=${5:-14}
proj_gamma=${6:-10.0}

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/sjkim/datasets/coco/val2014
else
  image_folder=/home/sjkim/datasets/gqa/images
fi

/home/sjkim/miniconda3/envs/vl/bin/python ./eval/object_hallucination_vqa_llava_steer.py \
  --model-path ${model_path} \
  --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
  --image-folder ${image_folder} \
  --answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_steer_L${proj_layer}_g${proj_gamma}_seed${seed}.jsonl \
  --directions_file ./diag_output/directions.pt \
  --proj_layer ${proj_layer} \
  --proj_gamma ${proj_gamma} \
  --seed ${seed}
