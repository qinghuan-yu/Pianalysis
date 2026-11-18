"""Stage the frozen Pianalysis listening probes locally. Never contacts a remote service."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data/auto_accompaniment/generalization_20261117"
TEMPLATES = ROOT / "examples/probe_release"
VARIANTS = ("melody", "block", "texture_42", "texture_7", "texture_2026")
PROTOCOL_HASH = "84abc86ebb9850f88ca5bf877fc5d8a9d46974f4fabfefee31b4f50f78dce471"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path, content):
    path.write_text(content, encoding="utf-8", newline="\n")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified_copy(source, destination, expected):
    shutil.copyfile(source, destination)
    if digest(destination) != expected:
        raise ValueError(f"Hash mismatch: {source.name}")


def player(path, label):
    return f'<label>{html.escape(label)}<audio controls preload="none" src="{path}"></audio></label>'


def build(output):
    dataset, blind = output / "hf_dataset", output / "blind_listening"
    for folder in (dataset / "audio", dataset / "midi", blind):
        folder.mkdir(parents=True)
    protocol_file = SUITE / "protocol.json"
    if digest(protocol_file) != PROTOCOL_HASH:
        raise ValueError("The frozen selection or protocol changed")
    protocol = read_json(protocol_file)
    report = read_json(SUITE / "report.json")
    probes = SUITE / "melodies.jsonl"
    verified_copy(probes, dataset / "melodies.jsonl", protocol["melodies_sha256"])
    rows = [json.loads(line) for line in probes.read_text(encoding="utf-8").splitlines()]
    focus = protocol["first_listen_case_ids"]
    ids = [row["song_id"] for row in rows]
    if len(rows) != 24 or len(set(ids)) != 24 or not set(focus) <= set(ids):
        raise ValueError("Unexpected probe inventory")
    metadata, sections, quick_links = [], {}, []
    for row in rows:
        sid, meta = row["song_id"], row["metadata"]
        if (not re.fullmatch(r"[a-z][a-z_]*", sid) or row["source"] != "assistant_authored_probe_20261117"
                or meta["training_use"] is not False or row["chords"] or row["accompaniment"]):
            raise ValueError("Expected an authored melody-only probe")
        audio = SUITE / "audio" / sid
        hashes = read_json(audio / "result.json")["artifact_sha256"]
        players = []
        for variant in VARIANTS:
            stem = f"{sid}_{variant}"
            wav, midi = stem + "_piano.wav", stem + ".mid"
            verified_copy(audio / wav, dataset / "audio" / wav, hashes[wav])
            verified_copy(audio / midi, dataset / "midi" / midi, hashes[midi])
            metadata.append(dict(file_name=wav, case_id=sid, title=meta["title"], variant=variant,
                                 focus=sid in focus, midi_path="midi/" + midi))
            label = {"melody": "原旋律", "block": "同和声 · 柱式对照"}.get(variant, variant.replace("texture_", "模型织体 · "))
            players.append(player("audio/" + wav, label))
        chords = " · ".join(f'{chord["start"]:g}: {chord["symbol"]}' for chord in report["cases"][sid]["harmony"]["chords"])
        sections[sid] = (f'<section><h2>{html.escape(meta["title"])} <small>{sid}</small></h2>'
                         f'<p>{html.escape(meta["description"])}</p><div class="players">{"".join(players[:3])}</div>'
                         f'<details><summary>另外两个种子与和弦起点（拍）</summary><p>{html.escape(chords)}</p>'
                         f'<div class="players">{"".join(players[3:])}</div></details></section>')
    for sid in focus:
        links = [f"[{label}](audio/{sid}_{variant}_piano.wav)" for variant, label in
                 (("melody", "Melody"), ("block", "Block chords"), ("texture_42", "Learned texture"))]
        quick_links.append(f'| {sid} | {" | ".join(links)} |')
    write_csv(dataset / "audio/metadata.csv", list(metadata[0]), metadata)
    card = (TEMPLATES / "dataset-card.md").read_text(encoding="utf-8")
    write_text(dataset / "README.md", card.replace("{{focus_table}}", "\n".join(quick_links)))
    shutil.copyfile(protocol_file, dataset / "protocol.json")
    for folder in (dataset, blind):
        shutil.copyfile(TEMPLATES / "MEDIA-LICENSE.md", folder / "MEDIA-LICENSE.md")
    layout = (TEMPLATES / "listening.html").read_text(encoding="utf-8")
    first = "".join(sections[sid] for sid in focus)
    rest = "".join(sections[sid] for sid in ids if sid not in focus)
    write_text(dataset / "index.html", layout.replace("{{title}}", "钢琴伴奏试听")
               .replace("{{content}}", "<p>先听推理前选定的六段。每段的柱式与织体使用同一和声计划；默认织体种子为 42。</p>"
                        + first + "<details><summary>其余 18 段（完整首轮）</summary>" + rest + "</details>"))

    rng = random.SystemRandom()
    order = rng.sample(focus, len(focus))
    assignments = ["block", "texture_42"] * 3
    rng.shuffle(assignments)
    answer_key, questions = [], []
    for index, (sid, first_variant) in enumerate(zip(order, assignments), 1):
        second_variant = "block" if first_variant == "texture_42" else "texture_42"
        number = f"{index:02}"
        for letter, variant in (("A", first_variant), ("B", second_variant)):
            shutil.copyfile(dataset / "audio" / f"{sid}_{variant}_piano.wav", blind / f"{number}{letter}.wav")
        answer_key.append(dict(pair=number, case_id=sid, A=first_variant, B=second_variant))
        fields = []
        for field, label in (("clarity", "哪个版本的旋律更清楚？"), ("texture", "哪个版本的伴奏更自然？")):
            fields.append(f'<label>{label}<select required name="{number}_{field}">'
                          '<option value="">请选择</option><option>A</option><option>B</option>'
                          '<option value="tie">无差别 / 无法判断</option></select></label>')
        questions.append(f'<section><h2>第 {number} 对</h2><div class="players">'
                         + player(number + "A.wav", "A") + player(number + "B.wav", "B")
                         + '</div><div class="players">' + "".join(fields) + '</div></section>')
    session = hashlib.sha256(json.dumps(answer_key).encode("utf-8")).hexdigest()[:12]
    form = ((TEMPLATES / "blind-form.html").read_text(encoding="utf-8")
            .replace("{{questions}}", "".join(questions)).replace("{{session}}", session))
    write_text(blind / "index.html", layout.replace("{{title}}", "六段钢琴盲听").replace("{{content}}", form))
    write_csv(output / "organizer_key.csv", ["session", "pair", "case_id", "A", "B"],
              [dict(session=session, **entry) for entry in answer_key])
    checksums = [f"{digest(path)}  {path.relative_to(dataset).as_posix()}" for path in sorted(dataset.rglob("*")) if path.is_file()]
    write_text(dataset / "SHA256SUMS", "\n".join(checksums) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new directory; never overwrites a previous blind session")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; use a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "candidate"
        build(staging)
        staging.rename(output)
    print(output)


if __name__ == "__main__":
    main()
