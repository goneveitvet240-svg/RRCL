# RRCL TMLR Synthetic v1 正式结果审计

> 正式运行日期：2026-08-07（Asia/Shanghai）
> 协议：`rrcl-tmlr-synthetic-v1`
> 判定：完整运行成功；所有预注册机制检查均按原门执行，未调幅度、网格、seed 或阈值，未重跑。

## 1. 冻结身份与产物

- freeze commit：`46a015c1af52c24cd722cfb5998df61596a2e161`；
- config SHA256：`7b9a9164ba827f0fefabde47162986a08c22f2a281d78e18ba52fa7b3489ad75`；
- formal provenance：`git_dirty=false`、`testing=false`；
- 规模：`16 cells × 10 seeds × 2 paired orders = 320 scenarios`；
- 正式目录：`runs_real/tmlr_synthetic_v1/`（由 `.gitignore` 排除）；
- manifest SHA256：`319316e5ed9ccc4be5d248d1786c851505fe132282e08c19e7d9d65bbf65ce27`；
- summary JSON SHA256：`ea5ad463c0a9783d20850d01b5fc7bdd955d4f402d35aa717b6f0eef5f10e4bd`；
- summary CSV SHA256：`e83227cea4e6f751dd65f6952a6d7d66e9b37a6b956915979c5162f1410b3ad1`；
- validator：`PASS TMLR synthetic-v1: raw hashes and derived summary/CSV reproduced`。

独立 supplement archive（补充归档）：

- 文件：`release/rrcl_tmlr_synthetic_v1_2026-08-07.zip`；
- 332 files（含归档内 `MANIFEST.json`），大小 1,344,771 bytes；
- SHA256：`4658d2bf4b44463e082643dba5653481b889d105e29c950b7db87b87bcf6ea1d`；
- 连续两次构建哈希一致，`unzip -t` 零错误；
- 本审计文档不进入归档，避免记录归档哈希造成 self-reference（自引用）。

`attempt.lock` 已永久保留。再次执行 formal runner 会被已有输出路径阻止；本结果是协议规定的唯一正式运行。

## 2. 预注册机制检查

| 检查 | 结果 | 判定 |
|---|---:|---|
| Null fixed-factor Oracle opportunity | `0.0341%`，上限 `0.1%` | PASS |
| normalized scale-isolation curve 最大差 | `0.0`，容差 `1e-8` | PASS |
| reliability precision direction | `20/20` scenarios 单调 | PASS |
| mapping-only material opportunity | `false` | 预注册负控制成立 |
| mapping + anisotropy material opportunity | `false` | 零结果保留，不调 covariance spectrum |

## 3. 主结果

所有 16 个 cell 的 fixed-factor Oracle opportunity 均低于预注册 `0.5%` 门：

- 最大值出现在 reliability-only cell：`0.200%`，95% paired-seed interval `[0.142%, 0.267%]`；
- 16/16 `opportunity_present=false`；
- 因而 16/16 `gap_confirmed=false`、16/16 `construction_candidate=false`；
- 四个 train-only rules 在 16 个 cell 中均没有 `useful=true`。

label-scale 开/关后的 normalized primary curves 完全一致，因此下表只列 scale-off 的 8 个唯一 normalized 机制组合。数字是相对 `f=1` 的 population balanced normalized-MSE gain（总体平衡归一化 MSE 增益，百分比；正值更好）。

| Cell | Reliability | Mapping | Anisotropy | Fixed-f Oracle | `f*_pred` | Precision | Shrinkage-`gamma` | VFF-RLS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0000 | 0 | 0 | 0 | +0.034 | +0.003 | -0.005 | +0.003 | +0.000 |
| 0001 | 0 | 0 | 1 | +0.050 | +0.005 | -0.001 | +0.001 | +0.000 |
| 0010 | 0 | 1 | 0 | +0.013 | -0.011 | -0.021 | +0.006 | -0.000 |
| 0011 | 0 | 1 | 1 | +0.019 | -0.089 | -0.031 | +0.002 | -0.069 |
| 0100 | 1 | 0 | 0 | +0.200 | +0.185 | +0.393 | +0.359 | +0.000 |
| 0101 | 1 | 0 | 1 | +0.179 | +0.117 | +0.341 | +0.290 | -0.000 |
| 0110 | 1 | 1 | 0 | +0.029 | -0.943 | -14.100 | -0.372 | -9.166 |
| 0111 | 1 | 1 | 1 | +0.014 | -0.001 | -4.542 | -0.171 | -6.704 |

## 4. 允许与禁止的解释

允许：

- “Under the preregistered balanced i.i.d. linear model, none of the 16 cells produced a material fixed-factor forgetting opportunity.”
- 已知 label scale 被 task-aware normalization 精确消除；reliability signal 的估计方向正确，但纯 reliability cell 的最好 deployable mean gain `0.393%` 仍低于 `0.5%` 门。
- mapping change、anisotropy 或二者组合在本模型中不是 material temporal forgetting opportunity 的充分条件。
- reliability weighting 与 mapping change 同时出现时，precision/VFF 可以严重伤害某些域，说明训练侧可靠性不是跨映射可交换的风险权重。

禁止：

- 不得写成“synthetic study 证明现实数据没有 drift”或“所有非均匀权重都无用”；
- 不得把 fixed-f Oracle 称为所有权重族的全局 Oracle；precision weighting 在 reliability-only cell 的均值可高于 fixed-f family；
- 不得因没有正 cell 修改 `0.5%` 门、增大 drift/covariance 幅度或另选 seed 重跑；
- 不得把 20 个 order rows 当 20 个独立样本；区间以 10 个 seed 为单位，先在 seed 内平均两个 order。

## 5. 对 TMLR 主张的影响

本结果没有在受控模型中复现 crowd development data 的大幅 Oracle opportunity，因此不提供该自然数据现象的唯一机制解释。它提供的是更窄但可验证的 boundary result（边界结果）：在均匀域风险、同分布 train/test 和预注册对称参数下，scale、reliability、mapping change 与 anisotropy 的简单开关不足以自动产生实质 temporal-forgetting opportunity；而 reliability-based rules 在 mapping change 存在时可发生明显负迁移。
