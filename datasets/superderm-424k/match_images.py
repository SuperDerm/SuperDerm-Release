#!/usr/bin/env python3
"""Link images you downloaded from a provider to the rows of SuperDerm-424K.

    python3 match_images.py --dataset SCIN --root /data/scin --out images/
    python3 match_images.py --dataset PAD-UFES-20 --root /data/pad-ufes-20 --out images/ --copy

Every file under --root is matched by SHA-256 first, then by the end of its path (the row's source_file).
Matches are written as <out>/<image_id><extension>: hard links by default, copies with --copy.
Rows that were not found are listed in <out>/missing-<dataset>.csv. Python standard library only.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default=str(Path(__file__).with_name("images.csv")), help="path to images.csv")
    parser.add_argument("--dataset", required=True, help="dataset name as written in images.csv, e.g. SCIN")
    parser.add_argument("--root", required=True, help="folder holding the files you downloaded from the provider")
    parser.add_argument("--out", required=True, help="folder to write <image_id><extension> files into")
    parser.add_argument("--copy", action="store_true", help="copy files instead of hard-linking them")
    args = parser.parse_args()

    with open(args.images, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == args.dataset]
    if not rows:
        sys.exit(f"No rows for dataset {args.dataset!r} in {args.images}.")
    by_hash = {row["sha256"]: row for row in rows if row["sha256"]}
    by_path = {row["source_file"]: row for row in rows}

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found, by_method = set(), {"sha256": 0, "path": 0}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        row, method = by_hash.get(file_sha256(path)), "sha256"
        if row is None:
            parts = path.relative_to(root).parts
            suffixes = ("/".join(parts[i:]) for i in range(len(parts)))
            row, method = next((by_path[s] for s in suffixes if s in by_path), None), "path"
        if row is None or row["image_id"] in found:
            continue
        target = out / f"{row['image_id']}{path.suffix.lower()}"
        if not target.exists():
            if args.copy:
                shutil.copy2(path, target)
            else:
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
        found.add(row["image_id"])
        by_method[method] += 1

    missing = [row for row in rows if row["image_id"] not in found]
    with open(out / f"missing-{args.dataset}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "source_file"])
        writer.writerows([row["image_id"], row["source_file"]] for row in missing)
    print(f"{args.dataset}: matched {len(found)} of {len(rows)} rows "
          f"({by_method['sha256']} by SHA-256, {by_method['path']} by path); {len(missing)} missing.")


if __name__ == "__main__":
    main()
