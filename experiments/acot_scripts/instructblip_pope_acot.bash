seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"adversarial"}
model_path=${4:-"/root/autodl-tmp/vicuna-7b-v1.1"}
cd_alpha1=${5:-3}
cd_beta=${6:-0.6}

if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/root/autodl-tmp/val2014
else
  image_folder=./data/gqa/images
fi

python ./eval/object_hallucination_vqa_instructblip_acot_pope.py \
--model-base ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/instructblip_results/instructblip_${dataset_name}_pope_${type}_answers_no_cd_seed${seed}_acot.jsonl \
--use_cd1 \
--use_cd2 \
--use_cd3 \
--cd_alpha1 $cd_alpha1 \
--cd_beta $cd_beta \
--seed ${seed}


