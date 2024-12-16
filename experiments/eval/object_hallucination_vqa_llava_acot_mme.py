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
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import nltk
from nltk import RegexpParser
from nltk.tokenize import word_tokenize
# import kornia
from transformers import set_seed
from acot_utils.acot_sample import evolve_acot_sampling
evolve_acot_sampling()
import pandas as pd
import spacy

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    tasks = ['existence','count','position','color']
    try:
        with open('./cache_dic1_mme.json','r') as f:
            cache_dic1 = json.load(f)
        with open('./cache_dic2_mme.json','r') as f:
            cache_dic2 = json.load(f)
        with open('./cache_dic3_mme.json','r') as f:
            cache_dic3 = json.load(f)
    except:
        cache_dic1 = {}
        cache_dic2 = {}
        cache_dic3 = {}
    data_path = 'data/MME_Benchmark/eval_tool/Your_Results'
    answers_path = f'data/MME_Benchmark/eval_tool/LLaVA15_Results_{args.seed}'
    
    qs__prompts={'existence':[f'{DEFAULT_IMAGE_TOKEN}\nWhat objects are in the image?',
                            f'{DEFAULT_IMAGE_TOKEN}\nWhat attributes do objects have in the image?',
                            f"{DEFAULT_IMAGE_TOKEN}\nWhat are the relationships between objects?"],
                 
                 'count':[f'{DEFAULT_IMAGE_TOKEN}\nWhat objects are in the image?',
                            f'{DEFAULT_IMAGE_TOKEN}\nWhat attributes do objects have in the image?',
                            f"{DEFAULT_IMAGE_TOKEN}\nWhat are the relationships between objects?"],
                 
                 'position':[f'{DEFAULT_IMAGE_TOKEN}\nWhat objects are in the image?',
                            f'{DEFAULT_IMAGE_TOKEN}\nWhat attributes do objects have in the image?',
                            f"{DEFAULT_IMAGE_TOKEN}\nWhat are the relationships between objects?"],
                 
                 'color':[f'{DEFAULT_IMAGE_TOKEN}\nWhat objects are in the image?',
                            f'{DEFAULT_IMAGE_TOKEN}\nWhat attributes do objects have in the image?',
                            f"{DEFAULT_IMAGE_TOKEN}\nWhat are the relationships between objects?"]}
    
    for task in tasks:
        print(task)
        with open(os.path.join(data_path, task+'.txt'),'r') as f:
            questions = f.readlines()
        answers_file = os.path.expanduser(os.path.join(answers_path,task+'.txt'))
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        for question in tqdm(questions):
            cur_prompt = question.split('\t')[1]
            image_file = question.split('\t')[0]
            image = Image.open(os.path.join(os.path.join(args.image_folder, task), image_file))
            image = image.resize((384,384))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            cache = task+image_file

            qs = cur_prompt

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            if args.use_cd1:
                conv_cd1 = conv_templates[args.conv_mode].copy()
                qs_ = qs__prompts[task][0]
                conv_cd1.append_message(conv_cd1.roles[0], qs_)
                conv_cd1.append_message(conv_cd1.roles[1], None)
                if cache_dic1.get(cache,None) == None:
                    prompt_ = conv_cd1.get_prompt()
                    input_ids_ = tokenizer_image_token(prompt_, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids_,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            image_sizes=[image.size],
                            do_sample=False,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_new_tokens=1024,
                            num_beams = 2,
                            use_cache=True)
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    cache_dic1[cache] = outputs
                else:
                    outputs = cache_dic1.get(cache,None)
                    
                conv_cd1.messages[-1][-1] = outputs
                conv_cd1.append_message(conv_cd1.roles[0], cur_prompt)
                conv_cd1.append_message(conv.roles[1], None)
                prompt_cd1 = conv_cd1.get_prompt()
                input_ids_cd1 = tokenizer_image_token(prompt_cd1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            else:
                input_ids_cd1 = None

            if args.use_cd2:
                conv_cd2 = conv_templates[args.conv_mode].copy()
                qs_ = qs__prompts[task][1]
                conv_cd2.append_message(conv_cd2.roles[0], qs_)
                conv_cd2.append_message(conv_cd2.roles[1], None)
                if cache_dic2.get(cache,None) == None:
                    prompt_ = conv_cd2.get_prompt()
                    input_ids_ = tokenizer_image_token(prompt_, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids_,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            image_sizes=[image.size],
                            do_sample=False,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_new_tokens=1024,
                            num_beams = 2,
                            use_cache=True)
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    cache_dic2[cache] = outputs
                else:
                    outputs = cache_dic2.get(cache,None)
                    
                conv_cd2.messages[-1][-1] = outputs
                conv_cd2.append_message(conv_cd2.roles[0], cur_prompt)
                conv_cd2.append_message(conv.roles[1], None)
                prompt_cd2 = conv_cd2.get_prompt()
                input_ids_cd2 = tokenizer_image_token(prompt_cd2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            else:
                input_ids_cd2 = None

            if args.use_cd3:
                conv_cd3 = conv_templates[args.conv_mode].copy()
                qs_ = qs__prompts[task][2]
                conv_cd3.append_message(conv_cd3.roles[0], qs_)
                conv_cd3.append_message(conv_cd3.roles[1], None)
                if cache_dic3.get(cache,None) == None:
                    prompt_ = conv_cd3.get_prompt()
                    input_ids_ = tokenizer_image_token(prompt_, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids_,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            image_sizes=[image.size],
                            do_sample=False,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_new_tokens=1024,
                            num_beams = 2,
                            use_cache=True)
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    cache_dic3[cache] = outputs
                else:
                    outputs = cache_dic3.get(cache,None)
                    
                conv_cd3.messages[-1][-1] = outputs
                conv_cd3.append_message(conv_cd3.roles[0], cur_prompt)
                conv_cd3.append_message(conv.roles[1], None)
                prompt_cd3 = conv_cd3.get_prompt()
                input_ids_cd3 = tokenizer_image_token(prompt_cd3, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            else:
                input_ids_cd3 = None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
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

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            # print(prompt)
            if args.use_cd1:
                print(prompt_cd1)
            if args.use_cd2:
                print(prompt_cd2)
            if args.use_cd3:
                print(prompt_cd3)
            print(outputs)
            ans_id = shortuuid.uuid()
            ans_file.write(question.strip()+'\t'+outputs + "\n")
            ans_file.flush()
        ans_file.close()
        
    with open('cache_dic1_mme.json', 'w') as file:
        json_data = json.dumps(cache_dic1)
        file.write(json_data)
    with open('cache_dic2_mme.json', 'w') as file:
        json_data = json.dumps(cache_dic2)
        file.write(json_data)
    with open('cache_dic3_mme.json', 'w') as file:
        json_data = json.dumps(cache_dic3)
        file.write(json_data)
    

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