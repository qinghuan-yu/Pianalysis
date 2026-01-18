#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单生成脚本 - 从头生成音乐（无条件生成）
"""

import json
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer


def generate_music_from_scratch():
    """从头生成音乐"""
    
    # ==================== 配置 ====================
    MODEL_PATH = "./model_output/checkpoint-6000"  # 使用最佳 checkpoint
    OUTPUT_FILE = "./generated_from_scratch.json"
    
    # 生成参数
    NUM_SEQUENCES = 3          # 生成 3 首曲子
    MAX_LENGTH = 600           # 每首曲子的长度
    TEMPERATURE = 0.9          # 温度（0.8-0.9 保守，1.0-1.1 创造性）
    TOP_K = 50
    TOP_P = 0.95
    
    # ==================== 加载模型 ====================
    TOKENIZER_PATH = "./tokenizer"
    print(f"正在加载模型: {MODEL_PATH}")
    print(f"正在加载 tokenizer: {TOKENIZER_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ 模型加载完成 ({device})")
    
    # ==================== 生成音乐 ====================
    print(f"\n" + "="*60)
    print(f"开始生成音乐...")
    print(f"  - 生成数量: {NUM_SEQUENCES}")
    print(f"  - 每首长度: {MAX_LENGTH} tokens")
    print(f"  - Temperature: {TEMPERATURE}")
    print("="*60 + "\n")
    
    # 从 BOS token 开始生成
    input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            num_return_sequences=NUM_SEQUENCES,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    
    # 转换结果
    generated_sequences = []
    for i, output in enumerate(output_ids):
        tokens = output.cpu().tolist()
        
        # 移除特殊 token
        filtered_tokens = [t for t in tokens if t not in [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]]
        
        generated_sequences.append(filtered_tokens)
        print(f"  ✓ 生成第 {i+1} 首: {len(filtered_tokens)} tokens")
    
    # ==================== 保存结果 ====================
    output_data = {
        "model_checkpoint": MODEL_PATH,
        "generation_type": "unconditional (from scratch)",
        "generation_params": {
            "max_length": MAX_LENGTH,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
        },
        "num_sequences": len(generated_sequences),
        "sequences": generated_sequences,
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*60)
    print(f"✓ 生成完成！")
    print(f"  - 结果保存在: {OUTPUT_FILE}")
    print(f"\n💡 下一步:")
    print(f"  1. 用你的解码器将 token IDs 转换回 MIDI")
    print(f"  2. 播放 MIDI 文件听听效果")
    print(f"  3. 如果不满意，可以调整 temperature 重新生成")
    print("="*60)


if __name__ == "__main__":
    generate_music_from_scratch()
