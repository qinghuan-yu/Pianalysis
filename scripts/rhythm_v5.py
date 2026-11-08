"""Density-conditioned onsets, followed by durations with known next attacks."""

import rhythm_v4 as base

VERSION = "local_rhythm_v5"
ONSETS_PHASE, DETAILS_PHASE = 8, 9
DENSITY = base.VOCAB
PROFILES = ("sparse", "medium", "dense")
VOCAB = DENSITY + len(PROFILES)


def profile_for(frames):
    rate = sum(f["mask"].bit_count() for f in frames) * base.SUBDIVISIONS / len(frames)
    return "sparse" if rate <= 2 else "medium" if rate <= 3 else "dense"


def prompt(label, melody, ppq, previous, profile="medium"):
    if profile not in PROFILES:
        raise ValueError("Unknown density profile")
    return base.prompt(label, melody, ppq, previous)[:-1] + [DENSITY + PROFILES.index(profile), base.SEP]


def duration_limit(frames, step, lane):
    if not frames[step]["mask"] & (1 << lane):
        raise ValueError("A silent lane has no duration")
    return next((i - step for i in range(step + 1, len(frames)) if frames[i]["mask"] & (1 << lane)), len(frames) - step)


def training_sample(label, melody, ppq, previous, frames, identity):
    # 训练密度从参考提取；推理密度由调用者指定，不能读取测试伴奏。
    profile = profile_for(frames)
    prefix = prompt(label, melody, ppq, previous, profile)
    tokens, labels = list(prefix), [-100] * len(prefix)
    tokens.append(ONSETS_PHASE)
    labels.append(-100)
    for step, frame in enumerate(frames):
        tokens.extend([base.STEP + step, base.MASK + frame["mask"]])
        labels.extend([-100, base.MASK + frame["mask"]])
    tokens.append(DETAILS_PHASE)
    labels.append(-100)
    for step, frame in enumerate(frames):
        if not frame["mask"]:
            continue
        tokens.append(base.STEP + step)
        labels.append(-100)
        for lane in range(3):
            if frame["mask"] & (1 << lane):
                duration, velocity = frame["durations"][lane], frame["velocities"][lane]
                if not 1 <= duration <= duration_limit(frames, step, lane) or velocity not in range(4):
                    raise ValueError("Reference event overlaps its next same-lane onset")
                values = [base.DURATION + duration - 1, base.VELOCITY + velocity]
                tokens.extend(values)
                labels.extend(values)
    tokens.append(base.EOS)
    labels.append(base.EOS)
    if len(tokens) > 256:
        raise ValueError("V5 rhythm context exceeds the configured model")
    return {"tokens": tokens, "labels": labels, "prefix_length": len(prefix), "source_id": identity,
            "density_profile": profile, "melody_slice": [prefix.index(base.MELODY) + 1, len(prefix) - 2]}


def build_dataset(annotation):
    dataset = base.build_dataset(annotation)
    pieces = {p["song"]: p for p in annotation["pieces"]}
    for record in dataset["records"]:
        piece = pieces[record["song"]]
        record["sample"] = training_sample(record["label"], [n for n in piece["notes"] if n["role"] == "melody"],
                                            piece["ppq"], record["previous"], record["frames"], record["source_id"])
    dataset.update(version=VERSION, density_source="reference_only_during_training; caller_profile_at_inference")
    return dataset


def predict(model, label, melody, ppq, previous, profile="medium"):
    import torch
    if getattr(model.config, "pianalysis_protocol", None) != VERSION:
        raise ValueError("V5 rhythm model protocol mismatch")
    tokens = prompt(label, melody, ppq, previous, profile)
    frames = base.empty_frames(base.steps_for(label, ppq))

    def choose(first, count):
        logits = model(input_ids=torch.tensor([tokens], device=model.device)).logits[0, -1]
        value = int(logits[first:first + count].argmax())
        tokens.append(first + value)
        return value

    with torch.inference_mode():
        tokens.append(ONSETS_PHASE)
        for step, frame in enumerate(frames):
            tokens.append(base.STEP + step)
            frame["mask"] = choose(base.MASK, 8)
        tokens.append(DETAILS_PHASE)
        for step, frame in enumerate(frames):
            if not frame["mask"]:
                continue
            tokens.append(base.STEP + step)
            for lane in range(3):
                if frame["mask"] & (1 << lane):
                    frame["durations"][lane] = choose(base.DURATION, duration_limit(frames, step, lane)) + 1
                    frame["velocities"][lane] = choose(base.VELOCITY, 4)
    return frames
