# Pianalysis - Symbolic Music Generation

A conditional music generation model based on **GPT-2** architecture, designed to generate full polyphonic piano pieces from single-voice melodies.

## 📋 Overview

- **Task**: Conditioned Generation (Melody → Full Polyphony)
- **Model**: GPT-2 (6 layers, 8 heads, 512 dim)
- **Data Format**: JSON Token Sequences (Compound Token)
- **Tokenizer**: Custom WordLevel Tokenizer

## 🗂️ Project Structure

```
Pianalysis/
├── train.py              # Training script
├── generate.py           # Inference script
├── requirements.txt      # Dependencies
├── config.json          # Configuration file
├── data/                # Data directory (to be created)
│   ├── song1.json
│   ├── song2.json
│   └── ...
├── tokenizer/           # Tokenizer save directory (auto-created during training)
└── output/              # Model output directory (auto-created during training)
    ├── checkpoint-xxx/
    └── final_model/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your music data (JSON format) in the `./data` directory.

**Data Format Example** (`data/song1.json`):
```json
[
  "TIME_SHIFT_0",
  "NOTE_ON_60_80_MELODY",
  "TIME_SHIFT_10",
  "NOTE_OFF_60_MELODY",
  "NOTE_ON_45_70_ACCOMP",
  "TIME_SHIFT_10",
  "NOTE_OFF_45_ACCOMP"
]
```

### 3. Train Model

```bash
python train.py
```

**Training Process**:
1. Automatically train WordLevel Tokenizer (if not exists)
2. Build dataset (extract melody as condition)
3. Train GPT-2 model
4. Save final model to `./output/final_model`

### 4. Generate Music

```bash
python generate.py
```

**Input Methods**:
- Method 1: Read melody from `melody_input.json`
- Method 2: Modify example melody in `generate.py`

## ⚙️ Configuration

### Training Parameters (`train.py`)

```python
DATA_DIR = "./data"              # Data directory
MAX_LENGTH = 1024                # Maximum sequence length
BATCH_SIZE = 8                   # Batch size
EPOCHS = 10                      # Number of epochs
LEARNING_RATE = 5e-4             # Learning rate
```

### Generation Parameters (`generate.py`)

```python
TEMPERATURE = 1.0                # Temperature (higher = more random)
TOP_K = 50                       # Top-K sampling
TOP_P = 0.95                     # Top-P (nucleus) sampling
NUM_SEQUENCES = 3                # Number of sequences to generate
```

## 📊 Model Architecture

```python
GPT2Config(
    vocab_size=auto_calculated,  # Automatically determined from data
    n_positions=2048,            # Context window
    n_embd=512,                  # Embedding dimension
    n_layer=6,                   # Number of transformer layers
    n_head=8,                    # Number of attention heads
)
```

## 🔍 Data Processing Pipeline

### Input Construction

For each piece, the training input format is:

```
<BOS> [Melody Tokens] <SEP> [Full Sequence Tokens] <EOS>
```

**Example**:
- **Source (Melody)**: All tokens with `_MELODY` suffix + `TIME_SHIFT`
- **Target (Full)**: Complete original sequence (including Melody and Accompaniment)

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<BOS>` | Beginning of sequence |
| `<EOS>` | End of sequence |
| `<SEP>` | Separator between melody and full output |
| `<PAD>` | Padding |
| `<UNK>` | Unknown token |

## 📈 Training Monitoring

View training logs with TensorBoard:

```bash
tensorboard --logdir ./output/logs
```

## 💡 Tips

1. **Dataset Size**: Recommended to prepare at least 100+ songs for good results
2. **Sequence Length**: If your music token sequences are very long, increase `MAX_LENGTH` to 2048
3. **Overfitting**: If dataset is small, reduce model layers (e.g., `n_layer=4`)
4. **Generation Quality**: Adjust `temperature` parameter
   - Lower (0.7-0.9): More conservative, closer to training data
   - Higher (1.0-1.2): More creative, more random

## 📝 Notes

- Ensure consistent token format in JSON files
- Melody tokens must include `_MELODY` suffix
- Accompaniment tokens should include `_ACCOMP` suffix
- TIME_SHIFT tokens are included in both Source and Target

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License
