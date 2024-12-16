import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, './')
os.environ['HF_HUB_OFFLINE']='1'
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from PIL import Image
import math

# import kornia
from transformers import set_seed
from acot_utils.acot_sample_qwen2vl import evolve_acot_sampling
evolve_acot_sampling()
import pandas as pd
import spacy

def eval_model(args):
    # Model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    try:
        with open('./cache_dic1_chair_qwen2vl.json','r') as f:
            cache_dic1 = json.load(f)
        with open('./cache_dic2_chair_qwen2vl.json','r') as f:
            cache_dic2 = json.load(f)
        with open('./cache_dic3_chair_qwen2vl.json','r') as f:
            cache_dic3 = json.load(f)
    except:
        cache_dic1 = {}
        cache_dic2 = {}
        cache_dic3 = {}
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))

        cur_prompt = line["text"]
        
        cache = image_file
        qs = cur_prompt

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": qs}
                ]
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        if args.use_cd1:
            qs_ = f'What objects are in the image?'
            conversation_cd1 = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type":"text","text":qs_}
                    ]
                }
            ]
            if cache_dic1.get(cache,None) == None:
                prompt_ = processor.apply_chat_template(conversation_cd1, add_generation_prompt=True)
                inputs_ = processor(
                        text=[prompt_], images=[image], padding=True, return_tensors="pt"
                    )
                inputs_ = inputs_.to("cuda")
                with torch.inference_mode():
                    output_ids = model.generate(**inputs_, 
                        do_sample=False,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        temperature=0,
                        max_new_tokens=1024,
                        num_beams = 2,
                        use_cache=True)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_.input_ids, output_ids)
                ]
                outputs = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                cache_dic1[cache] = outputs
            else:
                outputs = cache_dic1.get(cache,None)
                
            conversation_cd1.append({"role": "assistant", "content": [{'text': outputs}]})
            conversation_cd1.append({"role": "user", "content":[{'text': qs}]})
            prompt_cd1 = processor.apply_chat_template(conversation_cd1, add_generation_prompt=True)
            input_ids_cd1 = processor(
                text=[prompt_cd1], images=[image], padding=True, return_tensors="pt"
            ).input_ids.to("cuda")
        else:
            input_ids_cd1 = None
        
        if args.use_cd2:
            qs_ = f'What attributes do objects have in the image?'
            conversation_cd2 = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type":"text","text":qs_}
                    ]
                }
            ]
            if cache_dic2.get(cache,None) == None:
                prompt_ = processor.apply_chat_template(conversation_cd2, add_generation_prompt=True)
                inputs_ = processor(
                        text=[prompt_], images=[image], padding=True, return_tensors="pt"
                    )
                inputs_ = inputs_.to("cuda")
                with torch.inference_mode():
                    output_ids = model.generate(**inputs_, 
                        do_sample=False,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        temperature=0,
                        max_new_tokens=1024,
                        num_beams = 2,
                        use_cache=True)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_.input_ids, output_ids)
                ]
                outputs = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                cache_dic2[cache] = outputs
            else:
                outputs = cache_dic2.get(cache,None)
                
            conversation_cd2.append({"role": "assistant", "content": [{'text': outputs}]})
            conversation_cd2.append({"role": "user", "content":[{'text': qs}]})
            prompt_cd2 = processor.apply_chat_template(conversation_cd2, add_generation_prompt=True)
            input_ids_cd2 = processor(
                text=[prompt_cd2], images=[image], padding=True, return_tensors="pt"
            ).input_ids.to("cuda")
        else:
            input_ids_cd2 = None
            
        if args.use_cd3:
            qs_ = f'What are the relationships between objects?'
            conversation_cd3 = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type":"text","text":qs_}
                    ]
                }
            ]
            if cache_dic3.get(cache,None) == None:
                prompt_ = processor.apply_chat_template(conversation_cd3, add_generation_prompt=True)
                inputs_ = processor(
                        text=[prompt_], images=[image], padding=True, return_tensors="pt"
                    )
                inputs_ = inputs_.to("cuda")
                with torch.inference_mode():
                    output_ids = model.generate(**inputs_, 
                        do_sample=False,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        temperature=0,
                        max_new_tokens=1024,
                        num_beams = 2,
                        use_cache=True)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_.input_ids, output_ids)
                ]
                outputs = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                cache_dic3[cache] = outputs
            else:
                outputs = cache_dic3.get(cache,None)
            
            conversation_cd3.append({"role": "assistant", "content": [{'text': outputs}]})
            conversation_cd3.append({"role": "user", "content":[{'text': qs}]})
            prompt_cd3 = processor.apply_chat_template(conversation_cd3, add_generation_prompt=True)
            input_ids_cd3 = processor(
                text=[prompt_cd3], images=[image], padding=True, return_tensors="pt"
            ).input_ids.to("cuda")
        else:
            input_ids_cd3 = None

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                input_ids_cd1=input_ids_cd1,
                input_ids_cd2=input_ids_cd2,
                input_ids_cd3=input_ids_cd3,
                cd_alpha1 = args.cd_alpha1,
                cd_beta = args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print(prompt)
        # if args.use_cd1:
        #     print(prompt_cd1)
        # if args.use_cd2:
        #     print(prompt_cd2)
        # if args.use_cd3:
        #     print(prompt_cd3)
        print(outputs)
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": 'qwen2vl',
                                   "image_id": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        

    with open('cache_dic1_chair_qwen2vl.json', 'w') as file:
        json_data = json.dumps(cache_dic1)
        file.write(json_data)
    with open('cache_dic2_chair_qwen2vl.json', 'w') as file:
        json_data = json.dumps(cache_dic2)
        file.write(json_data)
    with open('cache_dic3_chair_qwen2vl.json', 'w') as file:
        json_data = json.dumps(cache_dic3)
        file.write(json_data)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_cd1", action='store_true', default=False)
    parser.add_argument("--use_cd2", action='store_true', default=False)
    parser.add_argument("--use_cd3", action='store_true', default=False)
    parser.add_argument("--cd_alpha1", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=55)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)