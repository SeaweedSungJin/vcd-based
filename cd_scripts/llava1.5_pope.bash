seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"popular"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/sjkim/datasets/coco/val2014
else
  image_folder=/home/sjkim/datasets/gqa/images
fi

/home/sjkim/miniconda3/envs/vl/bin/python ./eval/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_vcd_seed${seed}.jsonl \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}


