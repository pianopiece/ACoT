import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
import spacy
import kornia
from lavis.models import load_model_and_preprocess
from acot_utils.acot_sample import evolve_acot_sampling
evolve_acot_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    try:
        with open('./cache_dic1_blip.json','r') as f:
            cache_dic1 = json.load(f)
        with open('./cache_dic2_blip.json','r') as f:
            cache_dic2 = json.load(f)
        with open('./cache_dic3_blip.json','r') as f:
            cache_dic3 = json.load(f)
    except:
        cache_dic1 = {}
        cache_dic2 = {}
        cache_dic3 = {}
        
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        prompt = qs +  " Please answer this question with one word."

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image = image.resize((384,384))
        image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
        cache = image_file
        if args.use_cd1:
            qs_ = f'What objects are in the image?'
            prompt_ = qs_
            if cache_dic1.get(cache,None) == None:
                with torch.inference_mode():
                    outputs = model.generate({"image": image_tensor, "prompt": prompt_},
                        use_nucleus_sampling=False, num_beams=2,
                        top_p = args.top_p, repetition_penalty=1.5)
                outputs = outputs[0]
                cache_dic1[cache] = outputs
            else:
                outputs = cache_dic1.get(cache,None)
                
            prompt_cd1 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: " + prompt_+"\nASSISTANT: "+outputs+ '</s>'+'\nUSER: '
            prompt_cd1 += prompt + '\nASSISTANT: '
        else:
            prompt_cd1 = None
            
        if args.use_cd2:
            qs_ = f'What attributes do objects have in the image?'
            prompt_ = qs_
            if cache_dic2.get(cache,None) == None:
                with torch.inference_mode():
                    outputs = model.generate({"image": image_tensor, "prompt": prompt_},
                        use_nucleus_sampling=False, num_beams=2,
                        top_p = args.top_p, repetition_penalty=1.5)
                outputs = outputs[0]
                cache_dic2[cache] = outputs
            else:
                outputs = cache_dic2.get(cache,None)
                
            prompt_cd2 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: " + prompt_+"\nASSISTANT: "+outputs+ '</s>'+'\nUSER: '
            prompt_cd2 += prompt + '\nASSISTANT: '
        else:
            prompt_cd2 = None
            
        if args.use_cd3:
            qs_ = f'What are the relationships between objects?'
            prompt_ = qs_
            if cache_dic3.get(cache,None) == None:
                with torch.inference_mode():
                    outputs = model.generate({"image": image_tensor, "prompt": prompt_},
                        use_nucleus_sampling=False, num_beams=2,
                        top_p = args.top_p, repetition_penalty=1.5)
                outputs = outputs[0]
                cache_dic3[cache] = outputs
            else:
                outputs = cache_dic3.get(cache,None)
                
            prompt_cd3 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: " + prompt_+"\nASSISTANT: "+outputs+ '</s>'+'\nUSER: '
            prompt_cd3 += prompt + '\nASSISTANT: '
        else:
            prompt_cd3 = None

        samples = {"image": image_tensor, "prompt": prompt,"prompt_cd1": prompt_cd1,"prompt_cd2": prompt_cd2,"prompt_cd3": prompt_cd3}
        samples = {sample:samples[sample] for sample in samples if samples[sample] != None}
        with torch.inference_mode():
            outputs = model.generate(samples, 
                                     use_nucleus_sampling=True, 
                                     num_beams=1, 
                                     top_p = args.top_p, 
                                     repetition_penalty=1, 
                                     temperature=args.temperature, 
                                     cd_alpha1 = args.cd_alpha1, 
                                     cd_beta = args.cd_beta)
        outputs = outputs[0]
        print(prompt)
        if args.use_cd1:
            print(prompt_cd1)
        if args.use_cd2:
            print(prompt_cd2)
        if args.use_cd3:
            print(prompt_cd3)
        print(outputs)
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "model_id": "instruct_blip",
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    with open('cache_dic1_blip.json', 'w') as file:
        json_data = json.dumps(cache_dic1)
        file.write(json_data)
    with open('cache_dic2_blip.json', 'w') as file:
        json_data = json.dumps(cache_dic2)
        file.write(json_data)
    with open('cache_dic3_blip.json', 'w') as file:
        json_data = json.dumps(cache_dic3)
        file.write(json_data)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--cd_beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    print(args.seed)
    eval_model(args)
