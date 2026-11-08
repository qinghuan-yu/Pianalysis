# Pianalysis

Melody-conditioned symbolic piano accompaniment generation.

The current pipeline is:

```text
MIDI dataset
-> enhanced Skyline + DP annotation OR named-track POP909 import
-> short conditional training windows with explicit note durations (TSD V2)
-> GPT-2 accompaniment training
-> windowed melody prompt + grammar-constrained accompaniment
-> MIDI export
```

## Current Status

TSD V2 is the current default: explicit note durations, grammar constraints,
windowed inference for melody MIDI, and a POP909 named-track importer. It uses a
local TSD-style codec inspired by MidiTok on the existing 10 ms grid. See
[UPSTREAM_ADOPTION.md](UPSTREAM_ADOPTION.md) for upstream references and usage,
and [TSD_V2_RESULTS.md](TSD_V2_RESULTS.md) for the 58-run comparison and limitations.

The constrained model improved raw held-out window coverage from 72.82% to
84.35%, with 16/16 syntactically valid outputs. This is a small controlled
experiment, not a listening-quality result. POP909 samples validated ingestion
but were not included in this training run.

V1/V1.1 checkpoints remain supported when supplied with matching inputs. See
[TOKEN_PROTOCOL_V1_1.md](TOKEN_PROTOCOL_V1_1.md) for the earlier protocol and
[HARMONY_V1.md](HARMONY_V1.md) for the optional harmony-conditioning workflow.

The next stage is a timed harmony annotation and block-chord audition baseline,
documented in [HARMONY_V2_BASELINE.md](HARMONY_V2_BASELINE.md). It annotates the
40 local pieces plus four POP909 samples and exports simple chord columns for
review. The current test melody uses the user's D-major IV–V–vi–iii progression.
These labels remain under review and have not been used for new model training.

The old `dataset_final.json` / `checkpoint-6000` flow has been replaced. The project now builds its own local dataset from the `MIDI/` folder and trains on:

```text
data/training_windows_tsd_v2/dataset_windows_v1.json
```

The training sequence format is:

```text
[BOS] source_melody [SEP] target_accompaniment [EOS]
```

Only `target_accompaniment` contributes to training loss. Source melody and padding labels are masked with `-100`.
The optional harmony workflow inserts `[HARMONY] harmony_plan` before `[SEP]`;
the current evaluated V2 model was trained without it.

## Main Files

```text
train_v2.py                         # Conditional GPT-2 training
generate_from_scratch.py            # Conditional accompaniment generation
scripts/closed_loop_v1.py           # MIDI/token/MIDI sanity check
scripts/dp_melody_cleaning_v1.py    # Enhanced Skyline + DP weak labeling
scripts/label_harmony_v1.py         # Rule-based harmony labeling
scripts/build_training_windows_v1.py # Short trainable windows
scripts/tokenizer_v2.py             # Explicit-duration TSD codec
scripts/duration_constraints.py     # Event grammar and time horizon
scripts/import_pop909.py            # Named-track import and song identities
scripts/evaluate_tsd_v2.py          # Paired representation/generation comparison
config.json                         # Reference defaults; CLI flags control runs
requirements.txt                    # Pinned Python dependencies
```

Generated local data is intentionally ignored by Git:

```text
MIDI/
data/
model_output/
test/
```

## Data Preparation

1. Put MIDI files in `MIDI/`.
2. Run weak melody/accompaniment annotation:

```powershell
python scripts/dp_melody_cleaning_v1.py --midi-dir MIDI --out-dir data/dp_cleaned_v1 --write-midi
```

3. Build trainable TSD windows:

```powershell
python scripts/build_training_windows_v1.py --annotated-dir data/dp_cleaned_v1/annotated_notes --out-dir data/training_windows_tsd_v2 --tokenizer-version pianalysis_tsd_v2 --max-length 1024 --window-seconds 8 --preview-limit 20
```

For exact reproduction of the V1.1 comparison, add
`--reference-windows data/training_windows_v1_1/dataset_windows_v1.json` to reuse
its sample IDs and time boundaries. Without it, the shorter representation may
produce different window counts.

Optional harmony labeling remains available; use a separate dataset/model
directory for this experiment and build windows from its `annotated_notes`:

```powershell
python scripts/label_harmony_v1.py --annotated-dir data/dp_cleaned_v1/annotated_notes --out-dir data/harmony_labeled_v1 --segment-seconds 2.0 --min-confidence 0.42
```

Current paired TSD dataset (using the reference windows):

```text
Processed pieces: 40
Accepted windows: 1530
Training / evaluation windows: 1306 / 224
Average window length: 443.63 tokens (V1.1: 573.84)
Maximum window length: 825 tokens
```

For POP909, use `scripts/import_pop909.py` instead of weak annotation; the
track mapping and commands are in [UPSTREAM_ADOPTION.md](UPSTREAM_ADOPTION.md).

## Training

Install dependencies in a working Python environment:

```bash
pip install -r requirements.txt
pip install accelerate
```

Train:

```bash
python train_v2.py --data-file data/training_windows_tsd_v2/dataset_windows_v1.json --output-dir model_output/accompaniment_gpt2_tsd_v2
```

Useful defaults:

```text
max_length: 1024
batch_size: 8
epochs: at most 20 (early stopping enabled)
learning_rate: 5e-4
eval split: by source_piece_id
early stopping patience: 3
```

Training writes:

```text
model_output/accompaniment_gpt2_tsd_v2/final_model/
model_output/accompaniment_gpt2_tsd_v2/training_metadata.json
model_output/accompaniment_gpt2_tsd_v2/train_results.json
model_output/accompaniment_gpt2_tsd_v2/eval_results.json
```

The already evaluated local model is in
`model_output/accompaniment_gpt2_tsd_v2_e20/final_model` and stopped at epoch
13.72. Fresh training defaults to a separate directory. Use `--model-path` to
select a newly trained model. Models and datasets are not bundled with Git.

## Generation

Generate accompaniment from a standalone melody MIDI:

```bash
python generate_from_scratch.py --melody-midi test/test1.mid --seed 7 --out-midi test/test1_tsd_v2.mid --out-json test/test1_tsd_v2.json
```

This defaults to the evaluated TSD V2 model, grammar constraints, the model's
training window length (8 seconds), and 384 new tokens per window. Dense inputs
are split further instead of truncating source tokens. Set `--window-seconds 0`
for a single pass or `--no-grammar-constraints` for an ablation. Windows are
independent; phrase continuity is not yet modeled.

When using a model trained with harmony prompts, pass a harmony JSON as well:

```bash
python generate_from_scratch.py --model-path model_output/accompaniment_gpt2_harmony/final_model --melody-midi test/test1.mid --harmony-json test/test1_harmony.json --window-seconds 0 --out-midi test/test1_harmony_generated.mid
```

Harmony prompts require an appropriately trained model and currently use
single-window generation; the TSD model above was not trained on harmony prompts.

Generate accompaniment from a melody source in the training-window dataset:

```bash
python generate_from_scratch.py --dataset data/training_windows_tsd_v2/dataset_windows_v1.json --sample-index 0 --out-midi generated_accompaniment.mid
```

Despite the historical filename, `generate_from_scratch.py` is now conditional. It uses:

```text
[BOS] source_melody [SEP]
```

or, when harmony tokens are present:

```text
[BOS] source_melody [HARMONY] harmony_plan [SEP]
```

as prompt, generates accompaniment tokens, decodes them to notes, and exports a MIDI containing original melody + generated accompaniment.

Generation also applies a conservative postprocess pass:

```text
pitch-range clamp
maximum note duration
minimum note duration
same-pitch overlap merge
maximum simultaneous accompaniment notes
melody-clearance filtering
truncate accompaniment after melody end
optional empty-bar bass fill
```

The JSON output includes a `sanitation_report` and a `generation_report`
(per window when windowed), including syntax diagnostics and forced EOS reasons.
The controlled evaluation measures raw notes before postprocessing.

## Verification

```powershell
python -m pip install -r requirements-harmony.txt
python -m unittest discover -s tests -v
python scripts/evaluate_tsd_v2.py --out-dir data/tsd_v2_audit/evaluation_new
```

There are 54 regression tests, including 22 harmony annotation/rendering tests.
The harmony dependencies are optional for TSD generation but required for the
full test suite. Evaluation requires the local V1.1 and V2
models/datasets and refuses to overwrite an existing summary.

## Important Caveats

- The current trained model uses weak labels from enhanced Skyline + DP; the POP909 importer also supports named-track labels.
- Priority manual-review files are listed in `DATA_CLEANING_DP_V1.md`.
- TSD V2 durations are implemented. The beat/bar-position extension in `TOKENIZER_V2.md` remains planned; it needs reliable beat alignment. V1, V1.1 and V2 require matching datasets and checkpoints.
- Syntax constraints do not guarantee musical quality. The current evidence covers one training seed, four held-out pieces and one long melody, without subjective listening evaluation.
- `vue-piano` should remain a separate annotation UI project. This repo should only keep shared schemas and data-processing/training code.
