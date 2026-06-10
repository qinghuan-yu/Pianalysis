# Data Cleaning DP V1

## Feasibility

Enhanced Skyline + dynamic programming is feasible for this project and is now implemented locally.

It is the right first engineering step because:

- It needs no new model training.
- It can process the current `MIDI/` folder immediately.
- It produces melody/accompaniment labels compatible with the existing conditional token format.
- It gives reviewable MIDI outputs for manual correction.
- It is easier to debug than a learned melody classifier.

It should still be treated as weak labeling, not final ground truth. The output is good enough for pipeline validation and a first training experiment, but not good enough to become the final dataset without listening/review.

## Implemented Script

```text
scripts/dp_melody_cleaning_v1.py
```

The script applies:

```text
MIDI
-> note extraction
-> top-k onset candidates
-> local melody score
-> dynamic programming path selection
-> melody/accompaniment role assignment
-> conditional token dataset
-> melody/accompaniment/roundtrip MIDI exports
```

The generated training format is:

```text
[BOS] source_melody [SEP] target_accompaniment [EOS]
```

The target is accompaniment-only, so the future training loss should ignore everything before `target_start_index`.

## Run

```powershell
& 'C:\Users\QingYu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\dp_melody_cleaning_v1.py --midi-dir MIDI --out-dir data\dp_cleaned_v1 --write-midi
```

## Outputs

```text
data/dp_cleaned_v1/dataset_dp_v1.json
data/dp_cleaned_v1/cleaning_report.json
data/dp_cleaned_v1/annotated_notes/*.json
data/dp_cleaned_v1/melody_midi/*_melody.mid
data/dp_cleaned_v1/accompaniment_midi/*_accompaniment.mid
data/dp_cleaned_v1/annotated_midi/*_annotated.mid
data/dp_cleaned_v1/roundtrip_midi/*_roundtrip.mid
```

These outputs are intentionally ignored by Git because they are generated data.

## Latest Dataset Result

Using the current local `MIDI/` folder:

- Processed: 40 MIDI files
- Failed: 0
- Average melody ratio: 38.33%
- Minimum melody ratio: 12.86%
- Maximum melody ratio: 58.42%
- Average sequence length: 20966 tokens
- Maximum sequence length: 48859 tokens

Priority manual-review items:

```text
High melody ratio:
- call-of-silence: 50.10%
- in-the-pool: 53.67%
- uchiage-hanabi: 58.42%

Low melody ratio:
- only-my-railgun: 12.86%
```

## Engineering Assessment

### What Works

- The pipeline can batch-process the full MIDI folder.
- The DP annotator produces a complete melody/accompaniment split for every file.
- It generates reviewable `melody.mid` and `accompaniment.mid` files.
- It produces `target_start_index`, which is the key field needed for masked conditional training.

### What Is Still Weak

- The method still depends on heuristics.
- Fast ornamental notes may still be mislabeled.
- Octave melody handling is conservative and collapses some octave doublings.
- Dense right-hand textures may over-label melody.
- Sparse or very high-energy accompaniment may under-label melody.
- The token sequences are still too long for the current `train_v2.py` max length.

## Next Changes

1. Add bar/window slicing before training.
2. Update `train_v2.py` to use `target_start_index` and mask source/padding labels with `-100`.
3. Add vue-piano import/export for `annotated_notes/*.json`.
4. Listen to the priority review MIDI files and correct labels manually.
5. After 20-50 manually approved files exist, compare:

```text
raw Skyline
enhanced Skyline + DP
manual labels
```

Only then should a learned note classifier be considered.

