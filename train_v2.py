#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符号化音乐生成 - GPT-2 训练脚本（适配预处理好的数据集）
Conditioned Generation: Melody -> Full Polyphony
"""

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer


class MusicDataset(Dataset):
    """
    音乐生成数据集类
    直接使用预处理好的 training_sequence（已经是 token ID）
    """
    
    def __init__(
        self, 
        samples: list,
        pad_token_id: int,
        max_length: int = 1024
    ):
        self.samples = samples
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        
        print(f"数据集加载完成，共 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """返回样本的 token IDs"""
        sample = self.samples[idx]
        
        # 获取 training_sequence（已经是 token ID 的列表）
        token_ids = sample['training_sequence']
        
        # 截断或填充到 max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # 创建 attention_mask（1 表示真实 token，0 表示 padding）
        attention_mask = [1] * len(token_ids)
        
        # Padding
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # 转换为 tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # 对于 Language Modeling，labels 就是 input_ids
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def create_tokenizer(vocab_size: int, save_path: str):
    """
    创建一个简单的 tokenizer（因为数据已经是 token ID，只需要定义特殊 token）
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    
    # 创建一个空的 WordLevel tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
    
    # 构建词表：0-4 是特殊 token，5+ 是音乐 token
    vocab = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<SEP>": 3,
        "<UNK>": 4,
    }
    
    # 添加音乐 token（ID 从 5 开始）
    for i in range(5, vocab_size):
        vocab[f"TOKEN_{i}"] = i
    
    tokenizer.model = models.WordLevel(vocab=vocab, unk_token="<UNK>")
    
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
        model_max_length=2048,
    )
    
    wrapped_tokenizer.save_pretrained(save_path)
    
    return wrapped_tokenizer


def create_model(vocab_size: int) -> GPT2LMHeadModel:
    """创建 GPT-2 模型"""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=2048,      # 上下文窗口长度
        n_embd=512,            # 嵌入维度
        n_layer=6,             # Transformer 层数
        n_head=8,              # 注意力头数
        resid_pdrop=0.1,       # Dropout
        embd_pdrop=0.1,
        attn_pdrop=0.1,
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
    DATA_FILE = "./data/dataset_final.json"    # 数据文件
    TOKENIZER_PATH = "./tokenizer"             # Tokenizer 保存路径
    OUTPUT_DIR = "./model_output"              # 模型输出目录
    MAX_LENGTH = 1024                          # 最大序列长度
    
    BATCH_SIZE = 16                            # 训练批次大小（优化for RTX 4070 Super 12GB）
    EPOCHS = 10                                # 训练轮数
    LEARNING_RATE = 5e-4                       # 学习率
    WARMUP_STEPS = 500                         # Warmup 步数
    SAVE_STEPS = 1000                          # 保存检查点间隔
    LOGGING_STEPS = 10                         # 日志记录间隔（改小以便看到进度）
    
    # ==================== 1. 加载数据 ====================
    print("\n" + "="*60)
    print("加载数据集...")
    print("="*60)
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    all_samples = data['samples']
    
    print(f"✓ 数据集加载完成")
    print(f"  - 总样本数: {len(all_samples)}")
    print(f"  - 平均 Token 长度: {metadata['avg_token_length']}")
    print(f"  - 最大 Token 长度: {metadata['max_token_length']}")
    
    # 获取词表大小（找出最大的 token ID + 1）
    print("  - 正在扫描所有样本以确定词表大小...")
    max_token_id = 0
    for sample in all_samples:  # 检查所有样本以确保准确
        max_id = max(sample['training_sequence'])
        if max_id > max_token_id:
            max_token_id = max_id
    
    vocab_size = max_token_id + 1
    print(f"  - 词表大小: {vocab_size}")
    
    # ==================== 2. 创建/加载 Tokenizer ====================
    if not os.path.exists(os.path.join(TOKENIZER_PATH, "tokenizer_config.json")):
        print("\n创建 Tokenizer...")
        tokenizer = create_tokenizer(vocab_size, TOKENIZER_PATH)
    else:
        print("\n加载已有的 Tokenizer...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    
    print(f"✓ Tokenizer 就绪，词表大小: {tokenizer.vocab_size}")
    
    # ==================== 3. 分割数据集 ====================
    split_idx = int(len(all_samples) * 0.9)  # 90/10 分割
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    
    print(f"\n训练集: {len(train_samples)} 个样本")
    print(f"验证集: {len(eval_samples)} 个样本")
    
    train_dataset = MusicDataset(train_samples, tokenizer.pad_token_id, max_length=MAX_LENGTH)
    eval_dataset = MusicDataset(eval_samples, tokenizer.pad_token_id, max_length=MAX_LENGTH)
    
    # ==================== 4. 创建模型 ====================
    model = create_model(tokenizer.vocab_size)
    
    # 设置特殊 token ID
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # ==================== 5. Data Collator ====================
    # 由于我们已经在 Dataset 中处理了 padding，这里使用简单的 collator
    from transformers import default_data_collator
    data_collator = default_data_collator
    
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
        dataloader_num_workers=0,  # Windows 上设置为 0
        
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
