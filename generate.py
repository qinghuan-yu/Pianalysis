#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符号化音乐生成 - 推理脚本
使用训练好的模型进行音乐生成
"""

import json
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from typing import List


def load_model_and_tokenizer(model_path: str):
    """加载训练好的模型和 tokenizer"""
    print(f"加载模型: {model_path}")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ 模型加载完成，设备: {device}")
    return model, tokenizer, device


def generate_music(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    melody_tokens: List[str],
    max_length: int = 1024,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1
) -> List[List[str]]:
    """
    根据旋律生成完整的音乐
    
    Args:
        model: GPT-2 模型
        tokenizer: Tokenizer
        device: 设备
        melody_tokens: 旋律 token 列表
        max_length: 生成的最大长度
        temperature: 温度参数（越高越随机）
        top_k: Top-K 采样
        top_p: Top-P (nucleus) 采样
        num_return_sequences: 返回的序列数量
    
    Returns:
        生成的音乐 token 列表
    """
    # 构建输入: <BOS> [Melody] <SEP>
    prompt = [tokenizer.bos_token] + melody_tokens + [tokenizer.sep_token]
    prompt_str = " ".join(prompt)
    
    # Tokenize
    input_ids = tokenizer.encode(prompt_str, return_tensors='pt').to(device)
    
    print(f"输入旋律长度: {len(melody_tokens)} tokens")
    print(f"开始生成...")
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    
    # 解码
    results = []
    for i, output in enumerate(output_ids):
        # 转为字符串
        decoded = tokenizer.decode(output, skip_special_tokens=False)
        
        # 分割为 token 列表
        tokens = decoded.split()
        
        # 提取 <SEP> 之后、<EOS> 之前的内容
        try:
            sep_idx = tokens.index(tokenizer.sep_token)
            if tokenizer.eos_token in tokens:
                eos_idx = tokens.index(tokenizer.eos_token)
                generated_tokens = tokens[sep_idx+1:eos_idx]
            else:
                generated_tokens = tokens[sep_idx+1:]
        except ValueError:
            generated_tokens = tokens
        
        results.append(generated_tokens)
        print(f"  - 序列 {i+1}: 生成 {len(generated_tokens)} tokens")
    
    return results


def main():
    # ==================== 配置 ====================
    MODEL_PATH = "./output/final_model"    # 训练好的模型路径
    MELODY_FILE = "./melody_input.json"    # 输入的旋律文件（可选）
    OUTPUT_FILE = "./generated_music.json" # 输出文件
    
    # 生成参数
    MAX_LENGTH = 1024
    TEMPERATURE = 1.0
    TOP_K = 50
    TOP_P = 0.95
    NUM_SEQUENCES = 3  # 生成 3 个不同的版本
    
    # ==================== 加载模型 ====================
    model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
    
    # ==================== 准备旋律输入 ====================
    # 方式 1: 从文件加载
    try:
        with open(MELODY_FILE, 'r', encoding='utf-8') as f:
            full_sequence = json.load(f)
        
        # 提取旋律部分
        melody_tokens = [
            token for token in full_sequence 
            if '_MELODY' in token or token.startswith('TIME_SHIFT')
        ]
    except FileNotFoundError:
        # 方式 2: 手动输入示例旋律
        print(f"未找到 {MELODY_FILE}，使用示例旋律")
        melody_tokens = [
            "TIME_SHIFT_0",
            "NOTE_ON_60_80_MELODY",
            "TIME_SHIFT_10",
            "NOTE_OFF_60_MELODY",
            "NOTE_ON_62_80_MELODY",
            "TIME_SHIFT_10",
            "NOTE_OFF_62_MELODY",
            "NOTE_ON_64_80_MELODY",
            "TIME_SHIFT_20",
            "NOTE_OFF_64_MELODY",
        ]
    
    print("\n" + "="*60)
    print("输入旋律:")
    print(melody_tokens[:10], "..." if len(melody_tokens) > 10 else "")
    print("="*60 + "\n")
    
    # ==================== 生成音乐 ====================
    generated_sequences = generate_music(
        model=model,
        tokenizer=tokenizer,
        device=device,
        melody_tokens=melody_tokens,
        max_length=MAX_LENGTH,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        num_return_sequences=NUM_SEQUENCES
    )
    
    # ==================== 保存结果 ====================
    output_data = {
        "melody_input": melody_tokens,
        "generated_sequences": generated_sequences,
        "generation_params": {
            "max_length": MAX_LENGTH,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
        }
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"✓ 生成完成！")
    print(f"  - 生成了 {len(generated_sequences)} 个序列")
    print(f"  - 结果保存在: {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
