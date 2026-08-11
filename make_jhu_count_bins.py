import argparse
import json
import os
from pathlib import Path
import random


def count_points(gt_path):
    n = 0
    with open(gt_path) as f:
        for line in f:
            if len(line.strip().split()) >= 2:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default="domain_splits")
    ap.add_argument("--config-out", default="domains_jhu_count3.json")
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root)
    train_img = root / "train" / "images"
    train_gt = root / "train" / "gt"
    rows = []
    for img in sorted(train_img.glob("*.jpg")):
        gt = train_gt / f"{img.stem}.txt"
        if gt.exists():
            rows.append((img.stem, count_points(gt)))
    if len(rows) < 3:
        raise RuntimeError(f"Too few JHU train pairs under {root}")

    rows.sort(key=lambda x: x[1])
    n = len(rows)
    bins = {
        "JHU_low": rows[: n // 3],
        "JHU_mid": rows[n // 3: 2 * n // 3],
        "JHU_high": rows[2 * n // 3:],
    }

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = []
    rng = random.Random(a.seed)
    for name, items in bins.items():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * a.test_ratio)))
        test_items = items[:n_test]
        train_items = items[n_test:]

        train_include = out_dir / f"{name}_train.txt"
        test_include = out_dir / f"{name}_test.txt"
        with open(train_include, "w") as f:
            for stem, _ in train_items:
                f.write(stem + "\n")
        with open(test_include, "w") as f:
            for stem, _ in test_items:
                f.write(stem + "\n")
        counts = [c for _, c in items]
        print(
            f"{name}: train={len(train_items)} test={len(test_items)} "
            f"count_range=[{min(counts)}, {max(counts)}] "
            f"mean={sum(counts) / len(counts):.1f}"
        )
        domains.append({
            "name": name,
            "kind": "jhu",
            "root": str(root),
            "train_split": "train",
            "test_split": "train",
            "train_include_file": str(train_include),
            "test_include_file": str(test_include),
        })

    payload = {"domains": domains}
    with open(a.config_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {a.config_out}")


if __name__ == "__main__":
    main()
