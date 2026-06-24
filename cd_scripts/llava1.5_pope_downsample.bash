#!/bin/bash
# Downsample-CD 평가 스크립트
# VCD 가우시안 노이즈 대신 저해상도 다운샘플 이미지를 noisy branch로 사용.
# --noise_step 인자가 scale_factor 역할 (4 → 1/4 해상도, 기본값 4)
# 사용법: bash cd_scripts/llava1.5_pope_downsample.bash [seed] [dataset] [type] [model_path] [cd_alpha] [cd_beta] [scale_factor]

seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"popular"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
scale_factor=${7:-4}   # 4=1/4 해상도(84px), 8=1/8(42px), 더 클수록 더 흐릿

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/sjkim/datasets/coco/val2014
else
  image_folder=/home/sjkim/datasets/gqa/images
fi

/home/sjkim/miniconda3/envs/vl/bin/python ./eval/object_hallucination_vqa_llava_downsample.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_downsample${scale_factor}_seed${seed}.jsonl \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $scale_factor \
--seed ${seed}
