# Pianalysis — research and reproduction notes

**Question.** When an accompaniment sounds wrong, is the chord plan wrong or is the piano writing wrong? Predicting timed chords first makes that distinction audible. Hold the melody and predicted chords fixed; compare a deterministic block-chord rendering with the learned texture. This isolates the effect of changing the renderer, not the value of fine-tuning versus the original upstream model.

**What is adapted here.** ReaLchords and Compose & Embellish supply the pretrained backbones. This project adapts their training targets and timing/vocabulary interface, separates conditioning notes from texture targets, groups works before splitting, and checks whether the output preserves the input melody. The 4/4 probes cover eight bars and use supplied bar origins; they do not test automatic pickup detection or long-form composition.

## What was trained

| Stage | Selected step | Microbatch × accumulation | Precision | Recorded best validation NLL | Checkpoint SHA256 |
|---|---:|---:|---|---:|---|
| Harmony | 2488 | 1 × 16 | fp32 | 0.311922 | `b97fd68670bcdd67f0425f766b172b2c4af543a6e2c272a2982d57fb868e64e2` |
| Texture | 1900 | 1 × 4 | float32 | 1.311198 | `3157b8ffc67737ffc29ce303217e5925db2d43cf028e90916b40b5e606702870` |

The losses are teacher-forced and not comparable across tasks. Harmony's number is the best seen during training, not a fresh measurement at step 2488. Texture early-stopped at 2700. Full configs: [harmony](../auto_accompaniment/harmony_selected.json), [texture](../auto_accompaniment/texture_selected.json). See the [upstream revisions](../auto_accompaniment/model_sources.json). These are training settings, not a promise of bitwise reproducibility across hardware.

| Accepted samples | Train | Validation | Test |
|---|---:|---:|---:|
| POP909-CL | 657 | 80 | 74 |
| Hooktheory | 17,963 | 2,152 | 2,774 |
| Distinct work groups, both sources | 10,608 | 1,193 | 1,482 |

The [ID-only split list](../auto_accompaniment/accepted_splits.tsv) covers 23,700 accepted records, not slicing windows or quarantined works. To check it, regenerate the dataset using the root README, then run `python scripts/export_training_splits.py --output data/rebuilt_splits.tsv` and compare it with the published TSV. The exporter verifies the frozen sample SHA256 before writing. POP909 was also used in upstream harmony pretraining; grouped splits alone cannot establish unseen-work performance there.

## Run one probe

From the repository root, install the dependencies in the root README and fetch the pinned upstream files. Use **your own adapted checkpoints** from the training runs, not an unmodified texture base model. Replace the checkpoint arguments with their actual paths; use a fresh output directory.

```bash
python scripts/generate_auto_accompaniment.py --samples examples/auto_accompaniment/paper_kite.jsonl --song-id paper_kite --bars 8 --harmony-checkpoint model_output/harmony_run1/last.pt --texture-checkpoint model_output/texture_run1/last.pt --seed 42 --device cuda --output-dir data/demo_paper_kite
python scripts/evaluate_auto_accompaniment.py --reference examples/auto_accompaniment/paper_kite.jsonl --predictions data/demo_paper_kite/prediction.json --split test --output data/demo_paper_kite/metrics.json
```

Listen to `output.mid` and inspect `harmony.json`. With the selected weights, our local smoke run produced 124 accompaniment notes; MIDI SHA256: `8b2961bf382674a842aee60011780d71d44956bf6dcea9a4dbe8e886c99331f7`. Different weights need not match this hash. This melody-only probe has no reference harmony or accompaniment: inspect the per-song melody preservation and event checks, **not** the null reference-based scores. To evaluate a corpus, supply references and predictions for the same time spans; missing predictions count against coverage.

## Preparing the listening release

The tracked [three MP3 previews](../../README.md) are sufficient for a first listen. On the maintainer's checkout with the frozen local audio archive, run:

```bash
python scripts/build_probe_release.py --output data/research_release_v1
```

The output directory must not already exist. Only the following materials should change hands:

- **HF dataset:** `hf_dataset/` contains the bilingual navigation (English dataset card, Chinese offline page), 24 assistant-authored melody inputs, and all 120 WAV/MIDI pairs. The six focus cases were fixed before generation. AudioFolder metadata is provided for the Hub; the offline page also works without the Hub. Music is released under [CC BY-NC 4.0](MEDIA-LICENSE.md). Upload **only this directory**; do not upload the parent folder, third-party training data, checkpoints or backups.
- **Blind session:** send listeners only `blind_listening/` (the entire folder). Ask them not to visit the labelled demo first. The page compares block chords with seed 42, using balanced randomized A/B assignments. Each listener returns an anonymous CSV for clarity and naturalness separately; ties are allowed. Keep `organizer_key.csv` private, use the matching session ID, and decode only after collection. Report listener counts, per-case preferences and ties; these six purpose-built cases are not a population estimate. No independent votes have been collected by this workflow.
- **Existing evidence:** a single listener accepted the 24-case batch overall. There are no per-case or per-seed subjective scores. Mechanical integrity checks and NLL do not establish a listener preference.
