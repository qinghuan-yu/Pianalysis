#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符号化音乐生成 - GPT-2 训练脚本
Conditioned Generation: Melody -> Full Polyphony
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


class MusicDataset(Dataset):
    """
    音乐生成数据集类
    构建格式: <BOS> [Melody Tokens] <SEP> [Full Sequence] <EOS>
    """
    
    def __init__(
        self, 
        data_files: List[str], 
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"加载数据集，共 {len(data_files)} 个文件...")
        for file_path in data_files:
            self._load_file(file_path)
        
        print(f"数据集加载完成，共 {len(self.samples)} 个样本")
    
    def _load_file(self, file_path: str):
        """从 JSON 文件加载一首曲子"""
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        
        if not tokens or len(tokens) < 10:
            return  # 过滤太短的序列
        
        # 提取 Source: 所有 MELODY 相关的 token 和 TIME_SHIFT
        source_tokens = []
        for token in tokens:
            if '_MELODY' in token or token.startswith('TIME_SHIFT'):
                source_tokens.append(token)
        
        # Target: 完整序列
        target_tokens = tokens
        
        # 如果没有旋律音符，跳过这个样本
        if not source_tokens:
            return
        
        # 构建训练序列: <BOS> [Source] <SEP> [Target] <EOS>
        full_sequence = (
            [self.tokenizer.bos_token] + 
            source_tokens + 
            [self.tokenizer.sep_token] + 
            target_tokens + 
            [self.tokenizer.eos_token]
        )
        
        # 转换为字符串（用空格分隔，因为 WordLevel Tokenizer 需要）
        sequence_str = " ".join(full_sequence)
        
        self.samples.append(sequence_str)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """返回 tokenized 样本"""
        sequence_str = self.samples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sequence_str,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 对于 Language Modeling，labels 就是 input_ids
        # DataCollator 会自动处理 loss 计算时忽略 padding
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def build_tokenizer(data_files: List[str], save_path: str) -> PreTrainedTokenizerFast:
    """
    训练一个 WordLevel Tokenizer
    
    Args:
        data_files: JSON 数据文件列表
        save_path: Tokenizer 保存路径
    
    Returns:
        PreTrainedTokenizerFast: 训练好的 tokenizer
    """
    print("="*60)
    print("开始训练 Tokenizer...")
    print("="*60)
    
    # 特殊 Token
    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '<UNK>']
    
    # 创建 WordLevel Tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token='<UNK>'))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    
    # 配置 Trainer
    trainer = WordLevelTrainer(
        special_tokens=special_tokens,
        min_frequency=1  # 所有出现过的 token 都保留
    )
    
    # 准备训练数据（需要将所有 token 写成一行行的文本）
    temp_corpus_file = 'temp_corpus.txt'
    with open(temp_corpus_file, 'w', encoding='utf-8') as f:
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as jf:
                tokens = json.load(jf)
                # 每个 token 用空格分隔
                line = " ".join(tokens)
                f.write(line + "\n")
    
    # 训练 Tokenizer
    tokenizer.train([temp_corpus_file], trainer)
    
    # 删除临时文件
    os.remove(temp_corpus_file)
    
    # 保存
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    
    # 包装为 PreTrainedTokenizerFast
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token='<BOS>',
        eos_token='<EOS>',
        pad_token='<PAD>',
        sep_token='<SEP>',
        unk_token='<UNK>',
    )
    
    # 保存 wrapped tokenizer
    wrapped_tokenizer.save_pretrained(save_path)
    
    vocab_size = wrapped_tokenizer.vocab_size
    print(f"✓ Tokenizer 训练完成！")
    print(f"  - 词表大小: {vocab_size}")
    print(f"  - 保存路径: {save_path}")
    print("="*60)
    
    return wrapped_tokenizer


def create_model(vocab_size: int) -> GPT2LMHeadModel:
    """
    创建 GPT-2 模型
    
    Args:
        vocab_size: 词表大小
    
    Returns:
        GPT2LMHeadModel: GPT-2 模型
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=2048,      # 上下文窗口长度
        n_embd=512,            # 嵌入维度
        n_layer=6,             # Transformer 层数
        n_head=8,              # 注意力头数
        resid_pdrop=0.1,       # Dropout
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=None,     # 稍后设置
        eos_token_id=None,
        pad_token_id=None,
    )
    
    model = GPT2LMHeadModel(config)
    
    print("="*60)
    print("模型配置:")
    print(f"  - 词表大小: {vocab_size}")
    print(f"  - 上下文长度: {config.n_positions}")
    print(f"  - 嵌入维度: {config.n_embd}")
    print(f"  - Transformer 层数: {config.n_layer}")
    print(f"  - 注意力头数: {config.n_head}")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("="*60)
    
    return model


def main():
    # ==================== 配置参数 ====================
    DATA_DIR = "./data"                    # JSON 数据文件目录
    TOKENIZER_PATH = "./tokenizer"         # Tokenizer 保存路径
    OUTPUT_DIR = "./output"                # 模型输出目录
    MAX_LENGTH = 1024                      # 最大序列长度
    
    BATCH_SIZE = 8                         # 训练批次大小
    EPOCHS = 10                            # 训练轮数
    LEARNING_RATE = 5e-4                   # 学习率
    WARMUP_STEPS = 500                     # Warmup 步数
    SAVE_STEPS = 1000                      # 保存检查点间隔
    LOGGING_STEPS = 100                    # 日志记录间隔
    
    # ==================== 1. 加载数据文件 ====================
    data_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    
    if not data_files:
        raise ValueError(f"在 {DATA_DIR} 目录下没有找到 JSON 文件！")
    
    print(f"\n找到 {len(data_files)} 个 JSON 数据文件")
    
    # ==================== 2. 训练/加载 Tokenizer ====================
    if not os.path.exists(os.path.join(TOKENIZER_PATH, "tokenizer_config.json")):
        tokenizer = build_tokenizer(data_files, TOKENIZER_PATH)
    else:
        print("加载已有的 Tokenizer...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
        print(f"✓ Tokenizer 加载完成，词表大小: {tokenizer.vocab_size}")
    
    # ==================== 3. 创建数据集 ====================
    # 简单的 80/20 分割
    split_idx = int(len(data_files) * 0.8)
    train_files = data_files[:split_idx]
    eval_files = data_files[split_idx:]
    
    print(f"\n训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(eval_files)} 个文件")
    
    train_dataset = MusicDataset(train_files, tokenizer, max_length=MAX_LENGTH)
    eval_dataset = MusicDataset(eval_files, tokenizer, max_length=MAX_LENGTH)
    
    # ==================== 4. 创建模型 ====================
    model = create_model(tokenizer.vocab_size)
    
    # 设置特殊 token ID
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ==================== 5. Data Collator ====================
    # 用于处理批次的 padding，并自动设置 labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal Language Modeling，不是 Masked LM
    )
    
    # ==================== 6. 训练参数 ====================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps=LOGGING_STEPS,
        
        save_steps=SAVE_STEPS,
        save_total_limit=3,  # 只保留最近的 3 个 checkpoint
        
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        fp16=torch.cuda.is_available(),  # 如果有 GPU，使用混合精度训练
        dataloader_num_workers=4,
        
        report_to="tensorboard",  # 使用 TensorBoard 记录日志
    )
    
    # ==================== 7. 创建 Trainer ====================
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # ==================== 8. 开始训练 ====================
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # ==================== 9. 保存最终模型 ====================
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("\n" + "="*60)
    print(f"✓ 训练完成！")
    print(f"  - 最终模型保存在: {final_model_path}")
    print("="*60)


if __name__ == "__main__":
    main()
