# Training Windows V1

## Purpose

The full-song token sequences are too long for the current GPT-2 training setup. `train_v2.py` uses a `MAX_LENGTH` around 1024, while whole-song sequences are usually tens of thousands of tokens.

This step converts cleaned annotated notes into short training windows.

Input:

```text
data/dp_cleaned_v1/annotated_notes/*.json
```

Output:

```text
data/training_windows_v1/dataset_windows_v1.json
data/training_windows_v1/window_report.json
data/training_windows_v1/preview_midi/*.mid
```

## Script

```text
scripts/build_training_windows_v1.py
```

The script builds samples in this format:

```text
[BOS] source_melody [SEP] target_accompaniment [EOS]
```

Every sample includes:

```text
target_start_index
```

Training should set labels before `target_start_index` to `-100`, and should also set padding labels to `-100`.

## Run

```powershell
& 'C:\Users\QingYu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\build_training_windows_v1.py --annotated-dir data\dp_cleaned_v1\annotated_notes --out-dir data\training_windows_v1 --max-length 1024 --window-seconds 8 --preview-limit 20
```

## Latest Result

Using `data/dp_cleaned_v1/annotated_notes`:

- Processed pieces: 40
- Accepted windows: 1530
- Rejected windows: 8
- Average window length: 573 tokens
- Minimum window length: 17 tokens
- Maximum window length: 1023 tokens

Rejected windows were empty or lacked either source melody or target accompaniment. This is expected for sparse tails/endings.

## Why Not Keep vue-piano Here?

The `_external_vue_piano/` clone is useful as a reference and as the future target for annotation UI improvements, but it should not live inside this repository as tracked source.

Recommended boundary:

- `Pianalysis`: data cleaning, tokenization, dataset building, training, generation.
- `vue-piano`: visual annotation/review UI and MIDI preview/export.

Shared contracts should be copied or packaged deliberately:

- annotated note JSON schema
- token schema
- import/export endpoints

The full `_external_vue_piano/` folder is ignored by Git and can be deleted locally after its useful code has been ported or upstreamed.

## Next Step

Update `train_v2.py` to train from:

```text
data/training_windows_v1/dataset_windows_v1.json
```

Required training changes:

1. Use `target_start_index`.
2. Mask source labels with `-100`.
3. Mask padding labels with `-100`.
4. Split train/eval by `source_piece_id`, not random individual windows.
5. Save dataset metadata with the checkpoint.

