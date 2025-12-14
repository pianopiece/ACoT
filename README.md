# Mitigating Object Hallucination through Assembled Chain-of-Thought Reasoning
The official repo of paper [Mitigating Object Hallucination through Assembled Chain-of-Thought Reasoning](https://link.springer.com/chapter/10.1007/978-981-95-3346-6_4).

## Authors
Xinhao Wang, Xinyu Ma, Shengyong Ding, Lidia S. Chao, Derek F. Wong.

## Abstract
Recent years have witnessed a rapid development of Large Vision-Language Models (LVLMs), spurred by progress in Large Language Models (LLMs). However, a significant challenge has emerged: Object Hallucination, where the generated textual descriptions misalign with the factual visual content. This issue poses a critical safety risk in applications requiring high accuracy and hampers further progress in LVLM development. While previous research has partially addressed hallucinations from a visual standpoint, the language model capability within LVLMs remains underutilized. In this work, we introduce **A**ssembled **C**hain-**o**f-**T**hought (**A-CoT**), a novel training-free method designed to mitigate object hallucinations through effective Chain-of-Thought (CoT) reasoning. By using diverse Chain-of-Thought prompts to leverage knowledge in LVLMs and integrating them through assembly decoding, A-CoT combines original and CoT outputs, enabling multi-perspective reasoning, enhancing output robustness, and significantly advancing hallucination mitigation techniques. Experiments on multiple object hallucination benchmarks—POPE, CHAIR, and MME Hallucination Subset—demonstrate that A-CoT significantly reduces object hallucinations and outperforms baseline models.

<div align=center>
<img src="./image/method.jpg"/>
</div>

## Content
[1. Data](#data)

[2. Model](#model)

[3. Requirements](#requirements)

[4. Running](#running)

[5. Thanks](#thanks)

[6. Citation](#citation)

[7. Contact](#contact)

## Data
We conducted experiments using four public datasets:

[1. POPE](https://github.com/AoiDragon/POPE)

[2. MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

[3. CHAIR](https://github.com/yuezih/less-is-more)

POPE and CHAIR question data is gotten directly from [AGLA](https://github.com/Lackel/AGLA). You need to download the COCO_val2014 image files. For the MME dataset, you can request the data through the provided link.

## Model
We experimented with three LVLMs: [LLaVA](https://github.com/haotian-liu/LLaVA), [InstructBLIP](https://github.com/salesforce/LAVIS), and [Qwen2VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

## Requirements
Detailed dependencies are listed in environmnet_*.yml.

## Running
To run experiments on all experiment with LLaVA 1.5, InstructBLIP or Qwen2VL, use the following commands:
```
cd ACoT/experiments
acot_scripts/all_{model_name}_{benchmark}.bash
```
To evaluate model performance on POPE, use `all_evaluate_pope.py` in *output* folder.

To evaluate model performance on CHAIR, use `eval.sh` in CHAIR-eval. It's noted that you need to do preparation first following the guidance of CHAIR repositoriy.

For MME, evaluate model performance following the guidance of their MME repositoriy.

## Thanks
The logit adjustment framework (i.e., `sample.py`) is based on [VCD](https://github.com/DAMO-NLP-SG/VCD).

## Citation
If our paper or code is helpful to you, please consider citing our work:
```
@inproceedings{wang2025mitigating,
  title={Mitigating Object Hallucination Through Assembled Chain-of-Thought Reasoning},
  author={Wang, Xinhao and Ma, Xinyu and Ding, Shengyong and Chao, Lidia S and Wong, Derek F},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={48--61},
  year={2025},
  organization={Springer}
}
```

## Contact
If you have any questions, please email mc36507@umac.mo.
