#!/bin/bash
# Ablation-Projection Decoding (APD) 평가 스크립트
# diagnose_hallucination_direction.py의 d_ablation 방향을 중간 레이어에서 직교 투영 제거.
# VCD noisy branch 없음. 표준 generation + 레이어 hook.
#
# 사전 조건: diag_output/directions.pt 존재
#   (cd experiments && python diagnose_hallucination_direction.py 로 생성)
#
# 사용법:
#   bash cd_scripts/llava1.5_pope_ablation_proj.bash [seed] [dataset] [type] [model_path] [proj_layer] [proj_gamma]
#
# 예시 (레이어 14, gamma=1.0 — 완전 직교 투영):
#   bash cd_scripts/llava1.5_pope_ablation_proj.bash 55 coco popular
#
# 예시 (gamma=0.5 — 약한 투영):
#   bash cd_scripts/llava1.5_pope_ablation_proj.bash 55 coco popular liuhaotian/llava-v1.5-7b 14 0.5

seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"popular"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
proj_layer=${5:-14}
proj_gamma=${6:-1.0}

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/sjkim/datasets/coco/val2014
else
  image_folder=/home/sjkim/datasets/gqa/images
fi

/home/sjkim/miniconda3/envs/vl/bin/python ./eval/object_hallucination_vqa_llava_ablation_proj.py \
  --model-path ${model_path} \
  --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
  --image-folder ${image_folder} \
  --answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_ablation_proj_L${proj_layer}_g${proj_gamma}_seed${seed}.jsonl \
  --directions_file ./diag_output/directions.pt \
  --proj_layer ${proj_layer} \
  --proj_gamma ${proj_gamma} \
  --seed ${seed}
