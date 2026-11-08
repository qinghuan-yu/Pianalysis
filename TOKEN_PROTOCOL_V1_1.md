# V1.1 event protocol and controlled experiment

V1.1 fixes numeric token collisions while retaining V1's relative-time,
NOTE_ON/NOTE_OFF representation. It is a baseline for the later bar/position
Tokenizer V2, not an implementation of V2.

## Protocol

The version is `pianalysis_typed_v1_1`. Existing datasets/checkpoints use
`pianalysis_numeric_v1`; the older `pianalysis_numeric_v1_harmony_prompt` name
is also recognized as legacy V1.

| Meaning | Token IDs | Values |
| --- | --- | --- |
| Control tokens | Existing IDs within 0–32 | PAD/BOS/SEP/EOS/TIME, note roles, harmony markers |
| Pitch | 64–191 | MIDI 0–127 |
| Velocity | 192–318 | 1–127 |
| Time delta | 319–1118 | 1–800 ticks |
| Chord | 1119–1238 | 12 roots × 10 qualities |
| Bass | 1239–1366 | MIDI 0–127 |
| Unknown bass | 1367 | Separate from bass pitch zero |
| Confidence | 1368–1377 | Buckets 0–9 |

Time quantization remains 10 ms by default. Deltas larger than 800 ticks are
split into consecutive TIME events without losing elapsed time. No pitch,
velocity, time, or harmony parameter shares an ID with a control token or
another parameter type. The fixed vocabulary includes values absent from the
training split, so valid unseen input pitches remain encodable.

V1.1 requires a fresh checkpoint. Model configs save
`pianalysis_tokenizer_version` and `pianalysis_quantum_ms`, including in
intermediate checkpoints. Generation rejects source/model protocol or time
quantum mismatches. Harmony JSON containing labels can be re-encoded for the
selected model; pre-encoded harmony tokens must declare the matching version.

Legacy checkpoints remain usable. Their generation uses an event-aware
stopping criterion: a numeric `3` in a TIME, pitch, or velocity field is not
EOS. V1.1 can use ordinary EOS stopping because parameter IDs are disjoint.

## Build and train

Keep the original data and checkpoints for comparison. The following commands
write a separate melody-only dataset and model:

```powershell
python scripts/build_training_windows_v1.py --annotated-dir data/dp_cleaned_v1/annotated_notes --out-dir data/training_windows_v1_1 --tokenizer-version pianalysis_typed_v1_1 --max-length 1024 --window-seconds 8 --preview-limit 4
python train_v2.py --data-file data/training_windows_v1_1/dataset_windows_v1.json --output-dir model_output/accompaniment_gpt2_v1_1_e20 --epochs 20 --report-to none
```

Use a new output directory for another run. The experiment uses the existing
training settings: seed 42, 6 layers, 8 heads, embedding size 512, batch size 8,
learning rate 0.0005, and the same piece-level train/evaluation split.

The rebuilt local dataset has 1530 windows. All source/target event values,
window boundaries, target offsets, and sequence lengths match the original
dataset. There are 1306 training windows from 36 pieces and 224 validation
windows from 4 pieces. Evidence is saved in
`data/protocol_v1_1_audit/dataset_equivalence.json`.

Preview MIDI now decodes melody and accompaniment with separate time cursors;
both parts start on the same timeline.

## Generate and inspect

```powershell
python generate_from_scratch.py --model-path model_output/accompaniment_gpt2_v1_1_e20/final_model --melody-midi test/test1.mid --max-new-tokens 384 --temperature 0.7 --top-k 30 --top-p 0.9 --seed 7 --out-midi test/test1_v1_1.mid --out-json test/test1_v1_1.json
```

The JSON includes `generation_report` in addition to `sanitation_report`:

- `stop_reason`: `eos`, `max_new_tokens`, `context_limit`, or `unexpected_end`.
- Requested and effective budgets, prompt length, and EOS position.
- Unknown tokens, invalid parameter events, and incomplete events.
- Unclosed note-ons, orphan note-offs, and wrong-role note events.
- Total elapsed ticks in complete, valid events.

Generation caps its budget at the model's remaining context capacity. A prompt
that already fills the context is rejected with a clear error. This does not
implement long-song window stitching. The standalone 14.77-second example
remains outside the eight-second training distribution.

Typed decoding drops malformed events and attempts to resume at the next
control token; legacy decoding retains its existing permissive behavior.
Diagnostics are taken before note sanitation. A decoder repair is not evidence
that the model generated a correct event.

## Paired evaluation

```powershell
python scripts/evaluate_token_protocol_v1.py
```

The script compares three explicit conditions:

1. Original V1 checkpoint with the original token-wise EOS behavior.
2. The same V1 checkpoint with event-aware EOS stopping.
3. Retrained V1.1 checkpoint with disjoint parameter IDs.

It selects two deterministic windows from each of the four held-out pieces,
plus the standalone melody, and uses sampling seeds 7 and 42. Each condition
therefore produces 18 outputs, for 54 outputs total. Sampling and sanitation
settings are the same in all conditions. The script checks model piece splits
and paired source/target events before evaluating.

Per-run tokens, raw notes, sanitized notes, diagnostics, and listening MIDI
files are stored under `data/protocol_v1_1_audit/evaluation/`. Aggregate results
are in `summary.json`. Existing completed experiment results are not overwritten.

Coverage is the fraction of the melody's duration occupied by at least one
accompaniment note, measured before sanitation. Long sustained notes can raise
coverage without improving music. Use coverage together with hanging notes,
time overshoot, sanitation changes, and listening. The experiment has one
training seed and four held-out pieces. It does not establish perceptual
quality or generalization to a broad music collection. Token loss across the
two vocabularies is not a standalone quality ranking.

## Regression evidence

Run `python -m unittest discover -s tests -v`.

The first RED run reproduced two wrong EOS decisions: `[TIME, 3, ...]` stopped
at index 1, and a stream containing parameter `3` but no EOS was reported as
finished. The new versioned encoding and harmony APIs were initially absent.
Subsequent RED cases exposed missing pipeline integration, incorrect reporting
when typed EOS interrupts a partial event, and acceptance of mixed or mislabeled
training protocols.

The GREEN implementation adds typed parameter conversion, event-aware legacy
stopping, checkpoint metadata, dataset validation, and generation diagnostics.
Seventeen tests pass, including real tiny GPT-2 generation with deterministic
forced token sequences, MIDI preview alignment, and checkpoint save/reload.
Core event parsing and ID definitions are shared in `scripts/token_protocol_v1.py`;
there was no separate broad refactor. Tests were rerun after integration.
