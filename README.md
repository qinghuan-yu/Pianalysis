# Pianalysis

Melody-conditioned symbolic piano accompaniment generation.

The current pipeline is:

```text
MIDI dataset
-> enhanced Skyline + DP melody/accompaniment annotation
-> short conditional training windows
-> GPT-2 accompaniment training
-> melody prompt + generated accompaniment
-> MIDI export
```

## Current Status

The old `dataset_final.json` / `checkpoint-6000` flow has been replaced. The project now builds its own local dataset from the `MIDI/` folder and trains on:

```text
data/training_windows_v1/dataset_windows_v1.json
```

The training sequence format is:

```text
[BOS] source_melody [SEP] target_accompaniment [EOS]
```

Only `target_accompaniment` contributes to training loss. Source melody and padding labels are masked with `-100`.

## Main Files

```text
train_v2.py                         # Conditional GPT-2 training
generate_from_scratch.py            # Conditional accompaniment generation
scripts/closed_loop_v1.py           # MIDI/token/MIDI sanity check
scripts/dp_melody_cleaning_v1.py    # Enhanced Skyline + DP weak labeling
scripts/build_training_windows_v1.py # Short trainable windows
config.json                         # Current project defaults
requirements.txt                    # Pinned Python dependencies
```

Generated local data is intentionally ignored by Git:

```text
MIDI/
data/
model_output/
.deps/
```

## Data Preparation

1. Put MIDI files in `MIDI/`.
2. Run weak melody/accompaniment annotation:

```powershell
& 'C:\Users\QingYu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\dp_melody_cleaning_v1.py --midi-dir MIDI --out-dir data\dp_cleaned_v1 --write-midi
```

3. Build trainable windows:

```powershell
& 'C:\Users\QingYu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\build_training_windows_v1.py --annotated-dir data\dp_cleaned_v1\annotated_notes --out-dir data\training_windows_v1 --max-length 1024 --window-seconds 8 --preview-limit 20
```

Latest local result:

```text
Processed pieces: 40
Accepted windows: 1530
Rejected windows: 8
Average window length: 573 tokens
Maximum window length: 1023 tokens
```

## Training

Install dependencies in a working Python environment:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train_v2.py --data-file data/training_windows_v1/dataset_windows_v1.json --output-dir model_output/accompaniment_gpt2 --overwrite-output-dir
```

Useful defaults:

```text
max_length: 1024
batch_size: 8
epochs: 4
learning_rate: 5e-4
eval split: by source_piece_id
early stopping patience: 3
```

Training writes:

```text
model_output/accompaniment_gpt2/final_model/
model_output/accompaniment_gpt2/training_metadata.json
model_output/accompaniment_gpt2/train_results.json
model_output/accompaniment_gpt2/eval_results.json
```

## Generation

Generate accompaniment from a melody source in the training-window dataset:

```bash
python generate_from_scratch.py --model-path model_output/accompaniment_gpt2/final_model --dataset data/training_windows_v1/dataset_windows_v1.json --sample-index 0 --out-midi generated_accompaniment.mid
```

Despite the historical filename, `generate_from_scratch.py` is now conditional. It uses:

```text
[BOS] source_melody [SEP]
```

as prompt, generates accompaniment tokens, decodes them to notes, and exports a MIDI containing original melody + generated accompaniment.

## Important Caveats

- The current melody labels are weak labels from enhanced Skyline + DP.
- Priority manual-review files are listed in `DATA_CLEANING_DP_V1.md`.
- Token representation is now documented and reversible, but it is still a compact numeric event stream. A future version should move to a clearer symbolic vocabulary or compound tokens.
- `vue-piano` should remain a separate annotation UI project. This repo should only keep shared schemas and data-processing/training code.

