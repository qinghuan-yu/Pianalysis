# Closed Loop V1

This is the first local proof that the MIDI data path can run end to end:

```text
MIDI -> note JSON -> conditional tokens -> MIDI
```

The current target is accompaniment-only:

```text
[BOS] source_melody [SEP] target_accompaniment [EOS]
```

This matches the intended product flow: keep the input melody unchanged, generate accompaniment, then merge melody + accompaniment into a final MIDI.

## Run

Use the bundled Python runtime plus local dependencies in `.deps`:

```powershell
& 'C:\Users\QingYu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\closed_loop_v1.py --midi-dir MIDI --out-dir data\closed_loop_v1 --write-midi
```

## Outputs

```text
data/closed_loop_v1/dataset_v1.json
data/closed_loop_v1/roundtrip_report.json
data/closed_loop_v1/notes/*.json
data/closed_loop_v1/roundtrip_midi/*_roundtrip.mid
data/closed_loop_v1/target_accompaniment_midi/*_accompaniment.mid
```

## Latest Result

Using the local `MIDI/` folder:

- Processed: 40 MIDI files
- Failed: 0
- Average sequence length: 20719 tokens
- Max sequence length: 47613 tokens
- Round-trip note count matched for every processed file

## Important Limitations

- Melody labels still come from Skyline weak labeling.
- Skyline is not good enough as final ground truth for complex piano arrangements.
- Generated dataset sequences are far longer than the current GPT-2 `max_length=1024`.
- The next version must slice by bars or musical windows.
- Tempo/key/time-signature warnings from `pretty_midi` should become part of dataset quality reports.

## Next Step

Build `closed_loop_v2`:

- Bar-aware slicing.
- Manual annotation JSON import from `vue-piano`.
- Loss masking support in `train_v2.py`.
- Dataset validation before training.

