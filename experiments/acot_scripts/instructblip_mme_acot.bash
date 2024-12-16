seed=${1:-55}
model_path=${2:-"/root/autodl-tmp/vicuna-7b-v1.1"}
cd_alpha1=${3:-3}
cd_beta=${4:-0.2}

image_folder=/root/ACoT/experiments/data/MME_Benchmark
python ./eval/object_hallucination_vqa_instructblip_acot_mme.py \
--model-base ${model_path} \
--image-folder ${image_folder} \
--use_cd1 \
--use_cd2 \
--use_cd3 \
--cd_alpha1 $cd_alpha1 \
--cd_beta $cd_beta \
--seed ${seed}


