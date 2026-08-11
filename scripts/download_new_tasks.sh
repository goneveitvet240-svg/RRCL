#!/bin/bash
# 下载 IQA (KADID-10k) 和 Depth (NYU Depth V2) 数据集
# 在服务器上运行: bash download_new_tasks.sh

set -e
export HF_ENDPOINT=https://hf-mirror.com
DDIR=/root/autodl-tmp/datasets

# ===== KADID-10k (IQA) =====
mkdir -p $DDIR/KADID10k
cd $DDIR/KADID10k
# 方案A: 直接下载 (约1.3GB)
wget -c "https://datasets.activeloop.ai/kadid10k/kadid10k.zip" -O kadid10k.zip 2>/dev/null || \
python3 -c "
import urllib.request
url = 'http://database.mmsp-kn.de/kadid-10k-database.html'
print('请访问 http://database.mmsp-kn.de/kadid-10k-database.html 手动下载 images.zip 和 dmos.csv')
"
# 方案B: 用 pip install gdown + gdown
# gdown --id 1HWr0DYMYIR9Iixs8RGNaSX7D-S1RWLG6 -O kadid10k.zip

echo "KADID-10k 目录: $DDIR/KADID10k"

# ===== NYU Depth V2 (preprocessed PNG pairs) =====
mkdir -p $DDIR/NYU_Depth
cd $DDIR/NYU_Depth
# 方案A: 从 HuggingFace 镜像下载 (~4.7GB)
pip install datasets -q
python3 -c "
from datasets import load_dataset
import numpy as np
from PIL import Image
import os

print('Downloading NYU Depth V2 via HF mirror...')
ds = load_dataset('sayakpaul/nyu_depth_v2', split='train')
os.makedirs('/root/autodl-tmp/datasets/NYU_Depth/rgb', exist_ok=True)
os.makedirs('/root/autodl-tmp/datasets/NYU_Depth/depth', exist_ok=True)
for i, ex in enumerate(ds):
    img = ex['image']
    dep = ex['depth_map']
    stem = f'{i:05d}'
    img.save(f'/root/autodl-tmp/datasets/NYU_Depth/rgb/{stem}.png')
    dep_arr = np.array(dep)
    # convert to 16-bit mm
    dep_mm = (dep_arr * 1000).astype(np.uint16)
    Image.fromarray(dep_mm).save(f'/root/autodl-tmp/datasets/NYU_Depth/depth/{stem}.png')
    if i % 100 == 0:
        print(f'  {i}/{len(ds)}')
print('Done: NYU Depth V2 saved as RGB + depth PNG pairs')
"

echo "===== 下载完成 ====="
echo "KADID-10k: $DDIR/KADID10k"
echo "NYU Depth: $DDIR/NYU_Depth"
