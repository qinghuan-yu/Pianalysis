---
license: cc-by-nc-4.0
language:
- en
- zh
tags:
- music
- symbolic-music
- piano
- audio
size_categories:
- n<1K
configs:
- config_name: listening
  data_files:
  - split: test
    path:
    - audio/*.wav
    - audio/metadata.csv
---

# Pianalysis · 24 piano accompaniment probes

Listen to the same melody alone, with block chords, and with learned piano accompaniment. The last two use the same automatically predicted chords. These six cases were selected **before inference**, one from each diagnostic group.

| Case | Input | Baseline | Model (seed 42) |
|---|---|---|---|
{{focus_table}}

For the full offline listening page, download this folder and open `index.html` in a browser. No installation, network connection or model weights are needed.

**Contents.** 24 eight-bar, 4/4 melodies × five versions = 120 WAVs and 120 matching MIDI files. The three texture seeds are 42, 7 and 2026. The clips are rendered using FluidSynth 2.6.2 and [GeneralUser GS 2.0.3](https://www.schristiancollins.com/generaluser.php), with a common post-gain across versions of each melody. The bank itself is not included. File pairs are indexed in `audio/metadata.csv`; `melodies.jsonl` contains the input notes and author-supplied bar origins. `protocol.json` is the unchanged **pre-inference** plan, not a results report. Weights: ReaLchords fine-tune step 2488; Compose & Embellish fine-tune step 1900.

**Provenance and limits.** These are assistant-authored diagnostic melodies, not transcriptions of the training songs and not claimed as the applicant's hand compositions. They were not used for fine-tuning. No reference chord labels, keys or manual corrections were supplied to the models. The author supplied the bar origin, including for pickups. A single listener accepted the batch overall; there are no independent per-case ratings. This is not a representative benchmark or proof of zero overlap with upstream pretraining.

**Music license.** The released melodies, note events, MIDI and recordings may be shared and adapted for noncommercial purposes under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Credit **Pianalysis 清棫**, link the source and license, and indicate modifications when sharing. See the [media license and example credit](MEDIA-LICENSE.md). Commercial use requires separate permission.

No POP909-CL / Hooktheory training examples, SoundFont, model weights or optimizer states are distributed here. The media license covers only rights the project can grant and does not replace third-party terms or the code's MIT license.

[Training code, upstream sources and notices](https://github.com/qinghuan-yu/Pianalysis)
