# Harmony Labeling V1

Harmony V1 adds a weak harmonic plan before accompaniment training:

```text
[BOS] source_melody [HARMONY] harmony_plan [SEP] target_accompaniment [EOS]
```

The purpose is not perfect music-theory analysis. The purpose is to give the
texture model a coarse harmonic scaffold so it does not have to infer harmony,
bass, voicing, rhythm, and texture all at once.

## Pipeline

```text
data/dp_cleaned_v1/annotated_notes/*.json
-> scripts/label_harmony_v1.py
-> data/harmony_labeled_v1/annotated_notes/*.json
-> scripts/build_training_windows_v1.py
-> data/training_windows_v1/dataset_windows_v1.json
```

## Labeling Method

For each fixed time segment, defaulting to 2 seconds:

1. Collect accompaniment notes active in the segment.
2. Build weighted pitch-class counts from note overlap and velocity.
3. Estimate the bass from the lowest accompaniment pitch.
4. Score candidate chords across roots and qualities:
   - major
   - minor
   - diminished
   - augmented
   - sus2
   - sus4
   - dominant 7
   - major 7
   - minor 7
   - half-diminished 7
5. Add small bonuses for bass-root match and melody compatibility.
6. Emit `NC` when confidence is below the threshold.

## Token Format

Each harmony segment becomes:

```text
BAR CHORD_* BASS_* CONF_*
```

Example:

```text
BAR CHORD_C:maj BASS_48 CONF_7
```

These are prompt tokens only. Training loss is still applied only to the
accompaniment target after `SEP`.

## Current Limits

- Segment duration is fixed in seconds, not true musical bars.
- Chords are absolute labels, not key-aware roman numerals.
- The rule labeler depends on melody/accompaniment separation quality.
- Low-confidence labels become `NC`; this is intentional and safer than noisy
  chord supervision.

## Next Step

After a harmony-prompt model is trained, compare it against the non-harmony
model using the same melody MIDI prompts and inspect:

- same-pitch overlap count
- excess polyphony removals
- empty-bar fills
- accompaniment note density
- subjective harmonic fit
