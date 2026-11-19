"""Create a deterministic caption-matched pilot from the supplied archive."""
import argparse
import csv
import hashlib
import io
import random
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

SOURCES = ("real", "sd21", "sdxl", "sd3", "dalle3", "midjourney")
ROOT = "coco_indoor_all"
SEED = 180280


def caption_key(text):
    return " ".join(text.casefold().split())


def prepare(archive, output, train_count=100, test_count=50):
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        with bundle.open(f"{ROOT}/metadata.csv") as stream:
            rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
        development = [row for row in rows if row["split"] in ("train", "validation")]
        train_captions = {row["caption_id"] for row in development if row["split"] == "train"}
        train_texts = {caption_key(row["caption"]) for row in development if row["split"] == "train"}
        groups = defaultdict(dict)
        for row in development:
            if row["instance"] == "0":
                groups[row["split"], row["caption_id"]][row["generator"]] = row

        rng = random.Random(SEED)
        manifest = []
        seen_hashes = set()
        for original, split, count in (("train", "train", train_count),
                                       ("validation", "test", test_count)):
            candidates = sorted(key for key, views in groups.items()
                                if key[0] == original and set(views) == set(SOURCES))
            rng.shuffle(candidates)
            accepted = 0
            used_captions = set()
            for key in candidates:
                views = groups[key]
                caption = caption_key(views["real"]["caption"])
                if caption in used_captions:
                    continue
                if split == "test" and (key[1] in train_captions or caption in train_texts):
                    continue
                files = [bundle.read(f"{ROOT}/{views[source]['path']}") for source in SOURCES]
                hashes = {hashlib.sha256(data).digest() for data in files}
                if hashes & seen_hashes:
                    continue
                seen_hashes.update(hashes)
                used_captions.add(caption)
                for source, data in zip(SOURCES, files):
                    relative = Path("images") / source / f"{key[1]}.jpg"
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    manifest.append(dict(split=split, caption_id=key[1],
                                         source=source, image=relative.as_posix()))
                accepted += 1
                if accepted == count:
                    break
            if accepted != count:
                raise ValueError(f"Expected {count} {split} groups, found {accepted}")
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.output)
