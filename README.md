# Pianalysis - Symbolic Music Generation

A conditional music generation model based on **GPT-2** architecture, designed to generate full polyphonic piano pieces from single-voice melodies.

⚠️ **Current Status**: Training data format issues detected. Model needs to be retrained with corrected token format including proper timestamps.

## 📋 Overview

- **Task**: Conditioned Generation (Melody → Full Polyphony)
- **Model**: GPT-2 (6 layers, 8 heads, 512 dim)
- **Data Format**: JSON Token Sequences (Token IDs)
- **Best Checkpoint**: checkpoint-6000 (eval loss: 0.393)

## 🗂️ Project Structure

```
Pianalysis/
├── train_v2.py              # Training script
├── generate_from_scratch.py # Generation script (unconditional)
├── requirements.txt         # Dependencies
├── config.json             # Configuration reference
├── data/                   # Training data directory
│   └── dataset_final.json  # Main training dataset (56K samples)
├── test/                   # Test data
│   └── tokens_sliced.json  # Test samples
├── tokenizer/              # Tokenizer files
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── model_output/           # Model checkpoints
    ├── checkpoint-6000/    # Best model (recommended)
    └── logs/              # TensorBoard logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required**: PyTorch 2.5+, Transformers 4.57+, Tokenizers 0.22+, TensorBoard

### 2. Verify Data Format

Training data should be in `data/dataset_final.json`:

```json
{
  "metadata": {
    "total_samples": 56719
  },
  "samples": [
    {
      "training_sequence": [1, 0, 10, 60, 100, 11, 60, 500, 3]
    }
  ]
}
```

⚠️ **Known Issue**: Current training data only contains 15 unique tokens (0-83), missing proper timestamps. Data preprocessing needs to be fixed before retraining.

### 3. Train Model

```bash
python train_v2.py
```

**Recommended settings** (RTX 4070 Super 12GB):
- Batch size: 16
- Epochs: 3-4 (avoid overfitting)
- Learning rate: 5e-4
- Monitor: TensorBoard logs in `model_output/logs/`

### 4. Generate Music

```bash
python generate_from_scratch.py
```

Generates 3 sequences (600 tokens each) from checkpoint-6000.

Output: `generated_sequences.json`

## ⚙️ Configuration

### Model Architecture

```python
GPT2Config(
    vocab_size=801,             # Token vocabulary size
    n_positions=2048,           # Maximum context window
    n_embd=512,                 # Embedding dimension
    n_layer=6,                  # Transformer layers
    n_head=8,                   # Attention heads
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
```

**Model Size**: ~20.37M parameters

### Training Parameters (`train_v2.py`)

```python
BATCH_SIZE = 16                 # Batch size (optimized for RTX 4070 Super)
MAX_LENGTH = 1024               # Maximum sequence length
EPOCHS = 10                     # Number of epochs (reduce to 3-4 to avoid overfitting)
LEARNING_RATE = 5e-4            # Learning rate
LOGGING_STEPS = 10              # Log every N steps
dataloader_num_workers = 0      # Windows compatibility
```

### Generation Parameters (`generate_from_scratch.py`)

```python
NUM_SEQUENCES = 3               # Number of sequences to generate
MAX_LENGTH = 600                # Maximum tokens per sequence
TEMPERATURE = 0.9               # Sampling temperature
TOP_K = 50                      # Top-K sampling
TOP_P = 0.95                    # Nucleus sampling
```

## 📊 Training Results

**Best Checkpoint**: checkpoint-6000
- **Step**: 6000
- **Epoch**: 1.88
- **Eval Loss**: 0.393
- **Status**: Best model before overfitting

**Training History**:
- Epoch 0-2: Loss decreased from 5.7 → 0.4 (healthy learning)
- Epoch 2-10: Loss dropped to 0.05 (severe overfitting)
- **Recommendation**: Use checkpoint-6000, retrain for 3-4 epochs max

## 🔍 Data Format

### Token ID Mapping (Current)

```python
Special Tokens:
- BOS (Begin): 1
- EOS (End): 2  
- PAD (Padding): 0
- SEP (Separator): 3

Note Events:
- NOTE_ON: 10
- NOTE_OFF: 11
- Pitch range: 21-108 (MIDI pitch)

Timestamps:
- Milliseconds: 0, 100, 200, 500, 1000, etc.
```

### Expected Format (After Fix)

```json
{
  "training_sequence": [
    1,      // BOS
    0,      // Timestamp 0ms
    10,     // NOTE_ON
    60,     // Pitch (Middle C)
    500,    // Timestamp 500ms
    11,     // NOTE_OFF
    60,     // Pitch
    1000,   // Timestamp 1000ms
    3       // EOS or SEP
  ]
}
```

## 📈 Training Monitoring

View TensorBoard logs:

```bash
tensorboard --logdir ./model_output/logs
```

Monitor:
- **train_loss**: Should decrease steadily
- **eval_loss**: Should stabilize around 0.4-0.5 (if continues decreasing below 0.2, may be overfitting)

## 💡 Tips & Troubleshooting

### Training

1. **Overfitting Prevention**:
   - Reduce epochs to 3-4
   - Monitor eval_loss closely
   - Stop when eval_loss stops improving
   - Consider early stopping

2. **Windows Compatibility**:
   - Always use `dataloader_num_workers=0`
   - Avoids multiprocessing spawn deadlock

3. **Memory Usage** (RTX 4070 Super 12GB):
   - Batch size 16: ~10GB VRAM
   - Batch size 32: May cause OOM

### Generation

1. **Quality Control**:
   - Temperature 0.8-0.9: Balanced creativity
   - Temperature > 1.0: More random
   - Temperature < 0.8: More conservative

2. **Token Format Validation**:
   - Check for proper timestamps
   - Verify BOS/EOS tokens present
   - Ensure NOTE_ON followed by pitch

### Data Preprocessing

⚠️ **Critical Issue**: Current data missing timestamps
- Fix preprocessing to include millisecond timestamps
- Regenerate dataset_final.json
- Retrain model with corrected data

## 🐛 Known Issues

1. **Training data format**: Only 15 unique tokens (0-83), missing timestamps
2. **Generated output**: Lacks proper timestamp structure
3. **Vocab mismatch**: Model vocab (801) >> actual data range (0-83)

**Status**: Requires data preprocessing fix and model retraining

## 📝 License

MIT License - See [LICENSE](LICENSE)

## 📄 License

MIT License
