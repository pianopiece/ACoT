import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default='qwen2vl')
    parser.add_argument("--dataset-name", type=str, default="coco")
    parser.add_argument("--dataset-type", type=str, default="popular")
    parser.add_argument("--seed-num", type=str, default="55")

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    seed_num = args.seed_num

    ans_file = f'./qwen2vl_results/{model_name}_{dataset_name}_pope_{dataset_type}_answers_no_cd_seed{seed_num}_acot.jsonl'

    label_file = f'../data/POPE/{dataset_name}/{dataset_name}_pope_{dataset_type}.json'

    # open ground truth answers
    gt_files = [json.loads(q) for q in open(os.path.expanduser(label_file), "r")]

    # open generated answers
    gen_files = [json.loads(q) for q in open(os.path.expanduser(ans_file), "r")]

    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    total_questions = len(gt_files)
    yes_answers = 0

    # compare answers
    for index, line in enumerate(gt_files):
        idx = line["question_id"]
        gt_answer = line["label"]
        assert idx == gen_files[index]["question_id"]
        gen_answer = gen_files[index]["text"]
        # convert to lowercase
        gt_answer = gt_answer.lower()
        gen_answer = gen_answer.lower()
        # strip
        gt_answer = gt_answer.strip()
        gen_answer = gen_answer.strip()
        # pos = 'yes', neg = 'no'
        if gt_answer == 'yes':
            if 'yes' in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == 'no':
            if 'no' in gen_answer:
                true_neg += 1
            else:
                yes_answers += 1
                false_pos += 1
        else:
            print(f'Warning: unknown gt_answer: {gt_answer}')
            unknown += 1
    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / total_questions
    yes_proportion = yes_answers / total_questions
    unknown_prop = unknown / total_questions
    # report results
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'yes: {yes_proportion}')
    print(f'unknow: {unknown_prop}')