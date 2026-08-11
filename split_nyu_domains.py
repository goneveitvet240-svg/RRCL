"""把 NYU Depth V2 按文件索引切成3个domain (各~400张train)。
运行: python split_nyu_domains.py --nyu-dir /root/autodl-tmp/datasets/NYU_Depth
"""
import argparse, glob, os, random
ap = argparse.ArgumentParser()
ap.add_argument("--nyu-dir", default="/root/autodl-tmp/datasets/NYU_Depth")
ap.add_argument("--n-domains", type=int, default=3)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

rgb_files = sorted(glob.glob(os.path.join(a.nyu_dir, "rgb", "*.png")))
stems = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]
random.seed(a.seed); random.shuffle(stems)

n = len(stems); chunk = n // a.n_domains
split_dir = os.path.join(a.nyu_dir, "splits"); os.makedirs(split_dir, exist_ok=True)
for d in range(a.n_domains):
    chunk_stems = stems[d*chunk:(d+1)*chunk]
    tr = chunk_stems[:int(len(chunk_stems)*0.8)]
    te = chunk_stems[int(len(chunk_stems)*0.8):]
    open(os.path.join(split_dir, f"domain{d}_train.txt"), "w").write("\n".join(tr))
    open(os.path.join(split_dir, f"domain{d}_test.txt"), "w").write("\n".join(te))
    print(f"Domain {d}: train={len(tr)}, test={len(te)}")
print(f"Splits saved to {split_dir}")
