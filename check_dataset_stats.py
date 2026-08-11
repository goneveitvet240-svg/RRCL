"""
数据集统计核查脚本
用法：python3 check_dataset_stats.py --jhu /path/to/jhu --sha /path/to/sha --shb /path/to/shb
输出每个数据集的 train/test 图像数、人数均值、中位数、最大最小值
"""
import argparse
import glob
import os
import numpy as np


def jhu_counts(root, split):
    gt_dir = os.path.join(root, split, "gt")
    counts = []
    for f in sorted(glob.glob(os.path.join(gt_dir, "*.txt"))):
        with open(f) as fp:
            n = sum(1 for l in fp if l.strip())
        counts.append(n)
    return counts


def shanghai_counts(root, split):
    split_dir = "train_data" if split in {"train", "train_data"} else "test_data"
    gt_dir = os.path.join(root, split_dir, "ground-truth")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(root, split_dir, "ground_truth")
    counts = []
    for f in sorted(glob.glob(os.path.join(gt_dir, "*.mat"))):
        try:
            from scipy.io import loadmat
            mat = loadmat(f)
            if "image_info" in mat:
                pts = mat["image_info"][0, 0][0, 0][0]
            else:
                pts = None
                for key in ("annPoints", "points", "gt"):
                    if key in mat:
                        pts = mat[key]
                        break
            if pts is not None:
                counts.append(len(np.asarray(pts).reshape(-1, 2)))
        except Exception as e:
            print(f"  skip {f}: {e}")
    return counts


def print_stats(name, counts):
    if not counts:
        print(f"{name}: no data found")
        return
    arr = np.array(counts)
    print(f"{name}: n={len(arr)}, mean={arr.mean():.1f}, "
          f"median={np.median(arr):.1f}, "
          f"min={arr.min()}, max={arr.max()}, "
          f"std={arr.std():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jhu", default=None)
    parser.add_argument("--sha", default=None)
    parser.add_argument("--shb", default=None)
    args = parser.parse_args()

    if args.jhu:
        print("=== JHU-Crowd++ ===")
        print_stats("  train", jhu_counts(args.jhu, "train"))
        print_stats("  val  ", jhu_counts(args.jhu, "val"))
        print_stats("  test ", jhu_counts(args.jhu, "test"))

    if args.sha:
        print("=== ShanghaiTech Part A ===")
        print_stats("  train", shanghai_counts(args.sha, "train"))
        print_stats("  test ", shanghai_counts(args.sha, "test"))

    if args.shb:
        print("=== ShanghaiTech Part B ===")
        print_stats("  train", shanghai_counts(args.shb, "train"))
        print_stats("  test ", shanghai_counts(args.shb, "test"))
