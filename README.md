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

The user accepted test1's D-major IV–V–vi–iii block-chord baseline and pickup
timing. See [HARMONY_V2_BASELINE.md](HARMONY_V2_BASELINE.md). A separate timed
harmony model is now available: its paired held-out chord-tone ratio rose from
44.4% to 52.2%, while test1 improved only from 47.9% to 50.1%. It remains an
experimental model; corpus labels and generated texture have not passed listening
review. Protocol, 54 generation comparisons and commands are documented in
[HARMONY_CONDITIONING_V2.md](HARMONY_CONDITIONING_V2.md).

A narrower local voicing baseline now passes the user's test1 block-chord and
simple-arpeggio audition. It uses 24 source-pinned, symbolically reviewed POP909
segments, with song-level splits and explicitly synthetic curriculum targets.
The model predicts four pitches for the current chord; timing, dynamics and the
two simple patterns are provided by rules. See [TRUSTED_HARMONY_V3.md](TRUSTED_HARMONY_V3.md)
for the annotation scope, raw-pitch checks, listening acceptance and commands.

The next experiment freezes that V3 voicing model and learns upper-note rhythm,
duration and relative velocity from 67 reviewed POP909 segments (39 train,
19 validation, 9 test; four songs total). On the single held-out test song,
projected onset F1 improved from 0.382 to 0.715 over fixed arpeggios, while
duration accuracy and melody responsiveness remain unresolved. The user accepted
the new test1 rhythm as more suitable. See [LEARNED_RHYTHM_V4.md](LEARNED_RHYTHM_V4.md)
for the projection losses, checkpoint selection, artifact checks and acceptance.

The next bounded experiment expands to 12 POP909 songs / 195 reviewed segments
and compares unchanged V4 training with a density-conditioned V5 model that
predicts all onsets before durations. Expanded data improves duration accuracy;
V5 provides sparse/medium/dense control but regresses on some onset metrics.
The user preferred accepted V4 on test1, so neither new candidate is promoted.
See [RHYTHM_V5_RESULTS.md](RHYTHM_V5_RESULTS.md) for paired comparisons, limitations,
the recorded listening preference and the experimental generation CLI.

Following that preference, a test1 diagnosis now separates note removal,
matched-note duration changes and velocity changes, with shared audio gain.
V5 repeats one onset pattern in six of seven full spans and barely responds to
same-length rhythm-history changes in this case. No new model was trained or
promoted; see [NATURALNESS_AUDIT.md](NATURALNESS_AUDIT.md) for the controlled
auditions, prompt probes and their limits.
The user found all three primary local edits natural, so no single cause was
identified. The user then accepted the +1-octave accompaniment on test1;
`scripts/preview_accompaniment_register.py` preserves melody, timing and velocity.
Sampled Grand Piano previews are now available in `data/piano_review_v4/index.html`.
Run `python scripts/setup_piano_renderer.py` once to fetch the pinned portable
FluidSynth/SoundFont, then `python scripts/render_piano_preview.py --out-dir data/piano_review_new`
to render both registers with shared gain. The user accepted the sampled-piano
test1 preview; see `examples/harmony/piano_v4_acceptance.json`. The accepted MIDI
and historical additive-synth audio remain unchanged.

A newly authored eight-bar G-major melody now has a frozen V4 versus fixed-arpeggio
comparison at `data/original_melody_v4/index.html`, with both accompaniments raised
one octave and rendered through the same sampled piano. Reproduce it with
`python scripts/audition_original_melody_v4.py --out-dir data/original_melody_v4_new`.
The score is stored in `examples/harmony/original_melody_v4_probe.json`; it is a
listening probe, excluded from training, with no reference accompaniment or accuracy score.
The user accepted this new model-accompaniment preview ("这挺完美的"); the exact
media and checkpoint hashes are in `examples/harmony/original_melody_v4_acceptance.json`.
No preference over the fixed-arpeggio control was stated.

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
scripts/harmony_conditioning.py     # Timed harmony prompt codec and window alignment
scripts/evaluate_harmony_conditioning.py # Correct/shifted-chord generation comparison
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

For the experimental timed harmony model, pass the accepted tick-based plan:

```bash
python generate_from_scratch.py --model-path model_output/accompaniment_gpt2_harmony_timed_v2/final_model --melody-midi test/test1.mid --harmony-json data/harmony_v2_review/plans/test1_pickup_4563.json --seed 7 --no-fill-empty-bars --out-midi test/test1_harmony_generated.mid --out-json test/test1_harmony_generated.json
```

Timed harmony is clipped and rebased for each window using the source MIDI tempo
map. The model records a separate harmony protocol version and rejects missing or
mismatched conditions. The default melody-only TSD checkpoint remains unchanged.
The earlier V1 harmony format still uses the legacy single-window workflow.

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

There are 102 regression tests, including 22 harmony annotation/rendering tests,
nine timed-condition tests, 12 local-voicing/review-gate tests and 11 real-rhythm
projection/rendering tests, eight density-conditioned decoding tests and eight
counterfactual-edit/shared-gain tests.
The harmony dependencies are optional for TSD generation but required for the
full test suite. Evaluation requires the local V1.1 and V2
models/datasets and refuses to overwrite an existing summary.

## Important Caveats

- The current trained model uses weak labels from enhanced Skyline + DP; the POP909 importer also supports named-track labels.
- Priority manual-review files are listed in `DATA_CLEANING_DP_V1.md`.
- TSD V2 durations are implemented. The beat/bar-position extension in `TOKENIZER_V2.md` remains planned; it needs reliable beat alignment. V1, V1.1 and V2 require matching datasets and checkpoints.
- Syntax constraints do not guarantee musical quality. The current evidence covers one training seed, four held-out pieces and one long melody, without subjective listening evaluation.
- `vue-piano` should remain a separate annotation UI project. This repo should only keep shared schemas and data-processing/training code.
