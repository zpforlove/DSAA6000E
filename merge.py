# filename merge_lora.py
# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import AutoPeftModelForCausalLM
import time
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_dir", type=str, default='medllm-7B/final')
    parser.add_argument("--baichuan_model_dir", type=str, default='Baichuan2-7B-chat')
    parser.add_argument("--merged_model_dir", type=str, default='Baichuan2-7B-MedLLM-Merged')
    args = parser.parse_args()

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.lora_model_dir, use_fast=False, trust_remote_code=True,
                                              local_files_only=True)
    model = AutoPeftModelForCausalLM.from_pretrained(args.lora_model_dir, trust_remote_code=True, local_files_only=True)

    model.generation_config = GenerationConfig.from_pretrained(args.baichuan_model_dir, local_files_only=True)
    print('Load to CPU time:', time.time() - start)

    # 合并模型，并转换为float16
    start = time.time()
    model = model.merge_and_unload()
    model = model.half()
    print('Merge and half time:', time.time() - start)

    tokenizer.save_pretrained(args.merged_model_dir)
    model.save_pretrained(args.merged_model_dir)
