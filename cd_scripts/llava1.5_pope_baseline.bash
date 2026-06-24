seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"popular"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/home/sjkim/datasets/coco/val2014
else
  image_folder=/home/sjkim/datasets/gqa/images
fi

/home/sjkim/miniconda3/envs/vl/bin/python ./eval/object_hallucination_vqa_llava.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_baseline_seed${seed}.jsonl \
--seed ${seed}
