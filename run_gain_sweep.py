"""Quick check: can ONE global variance-gain align f*_pred to f*_oracle across ALL
crowd+age sequences? If no single column matches the oracle column, the isotropic
theory's misalignment is a rank error (not a scale constant) -> global gain can't fix.
Reuses run_theory_predict; loads features once per config (cached)."""
import json
import numpy as np
from run_theory_predict import load_task, estimate, f_pred, f_oracle

CFGS = [
    ("domains_jhu_sha_shb.json", "crowd"),
    ("domains_jhu_shb_sha.json", "crowd"),
    ("domains_sha_shb.json", "crowd"),
    ("domains_sha_shb_jhu.json", "crowd"),
    ("domains_qnrf_sha_shb.json", "crowd"),
    ("domains_qnrf_shb_sha.json", "crowd"),
    ("domains_sha_shb_qnrf.json", "crowd"),
    ("domains_age_utk_young_old.json", "age"),
    ("domains_age_agedb_utk.json", "age"),
    ("domains_age_utk_agedb.json", "age"),
]
GAINS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
BB, IMG, LAM, MPD = "vit_base_patch14_dinov2.lvd142m", 518, 100.0, 400
grid = np.round(np.arange(0.05, 1.0001, 0.05), 3)
forgets = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

hdr = "config".ljust(22) + " ".join(f"g{g:g}".rjust(6) for g in GAINS) + "  | oracle"
print(hdr); print("-" * len(hdr))
for cfgp, task in CFGS:
    try:
        cfg = json.load(open(cfgp))
        tr, te, _ = load_task(cfg, task, BB, IMG, MPD)
        pd, d_aug = estimate(tr, LAM)
        fo, _, _ = f_oracle(tr, te, forgets, LAM)
        fps = [f_pred(pd, d_aug, LAM, grid, gain=g)[0] for g in GAINS]
        print(cfgp.replace("domains_", "").replace(".json", "").ljust(22)
              + " ".join(f"{x:.2f}".rjust(6) for x in fps) + f"  |  {fo:.2f}")
    except Exception as e:
        print(cfgp.ljust(22) + f"  ERROR: {e}")
print("\n读法：找一列 gain，使该列各行都≈最右 oracle 列。找不到 → 排序错，全局常数救不了。")
