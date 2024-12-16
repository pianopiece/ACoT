model_name=${1:-"llava15"}
python chair.py \
--coco_path MSCOCO/annotations \
--cap_file ../${model_name}_chair_answers_no_cd_seed55_acot.jsonl \
--cache ./chair.pkl

python chair.py \
--coco_path MSCOCO/annotations \
--cap_file ../${model_name}_chair_answers_no_cd_seed42_acot.jsonl \
--cache ./chair.pkl

python chair.py \
--coco_path MSCOCO/annotations \
--cap_file ../${model_name}_chair_answers_no_cd_seed37_acot.jsonl \
--cache ./chair.pkl

