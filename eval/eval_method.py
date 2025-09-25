import os
import torch
import random
import numpy as np
import argparse
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import roc_auc_score, roc_curve

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_dataset():
    dataset = load_dataset("simplescaling/s1K_tokenized")
    dataset = dataset['train']
    train_test_split = dataset.train_test_split(train_size=0.8, seed=42, shuffle=True)
    train_dataset = train_test_split['train']
    test_dataset =  train_test_split['test']
    print(f"train dataset: {len(train_dataset)} test_dataset: {len(test_dataset)}")
    return train_dataset, test_dataset

def get_perplexity(logprobs):
    probs = []
    for logprob in logprobs:
        assert len(logprob) == 1, "logprob should be a list of length 1"
        for key, value in logprob.items():
            logprob_value = value.logprob
            probs.append(logprob_value)
    return probs


def token_probability_deviation(sample_pro, tau=1, alpha=0.6):
    tbd = []
    for i, pro in enumerate(sample_pro):
        dev = [tau - p for p in pro if p < tau]
        dev = np.array([0]) if len(dev) == 0 else np.array(dev)
        tbd_score = np.mean(dev ** alpha)
        tbd.append(tbd_score)
    return tbd

def get_metrics(sample_score, labels):
    scores = [-value for value in sample_score]
    auc = roc_auc_score(labels, scores)
    tpr_1fpr = 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    for i, fpr_value in enumerate(fpr):
        if fpr_value >= 0.01:
            tpr_1fpr = tpr[i]
            break
    return auc, tpr_1fpr
    

def eval(args):
    train_dataset, test_dataset = get_dataset()
    members = train_dataset.map(lambda example: {**example, "label": 1})
    non_members = test_dataset.map(lambda example: {**example, "label": 0})
    members = members.select(range(len(non_members)))
    data = concatenate_datasets([members, non_members])
    model = LLM(
        model = args.model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )
    tok = AutoTokenizer.from_pretrained(args.model_path)
    stop_token_ids = tok("<|im_end|>")["input_ids"]

    sampling_params = SamplingParams(
        max_tokens=1000,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        temperature=0,
        logprobs=0
    )
    
    total = 0
    batch_size = 16
    num_batches = math.ceil(len(data) / batch_size)
    data_list = []
    
    for bath_idx in range(num_batches):
        batch_data = data.select(range(bath_idx * batch_size, min((bath_idx + 1) * batch_size, len(data))))
        prompt_list = []
        for example in batch_data:
            question = example['question']
            prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
            prompt_list.append(prompt)
        
        outputs = model.generate(prompt_list, sampling_params=sampling_params)
        
        for i, example in enumerate(batch_data):
            response = outputs[i].outputs[0].text
            tokens_nums = len(outputs[i].outputs[0].token_ids)
            logprobs = outputs[i].outputs[0].logprobs
            probs = get_perplexity(logprobs) 
            example['response'] = response
            example['tokens_nums'] = tokens_nums
            example['logprobs'] = probs
            total += 1
            print(f"Finish {total} samples")
            data_list.append(example)

    token_nums = 300
    sample_logprob = [ex['logprobs'][:token_nums] for ex in data_list]
    sample_pro = [[math.exp(pro) for pro in logpros] for logpros in sample_logprob]
    sample_scores = token_probability_deviation(sample_pro)
    labels = [ex['label'] for ex in data_list]
    auc, tpr = get_metrics(sample_scores, labels)
    print(f"Our new method: {auc:.4f}   TPR at 0.01 FPR: {tpr:.4f}")
    
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")

    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser_args()
    seed_everything(42)
    eval(args)
    
    