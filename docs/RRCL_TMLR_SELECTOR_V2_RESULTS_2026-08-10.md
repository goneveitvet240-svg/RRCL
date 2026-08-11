# RRCL TMLR selector-v2 formal result

> 正式运行日期：2026-08-10（Asia/Shanghai）<br>
> 预结果 freeze commit：`522260e56d2f17d1d7b7c982812f55d184a647c9`<br>
> 配置 SHA-256：`5dbb78810f903291d0c1459401ee6c496d9226d4a53562bac25bd7a9b4affcbd`<br>
> 正式 result SHA-256：`329de9484f0ac337e543176834dc3f08437f0673b46643290c6aa3cba2f5d2a5`

## 一句话结论

learned KNN selector（学习型 KNN 选择器）在完全留出的 256 个任务上优于
constant `f=1`，并在 40 个 scalar-grid opportunity-present（冻结标量网格内机会存在）任务上获得
`+0.520% [0.330%, 0.728%]` utility；但它的 Oracle regret（Oracle 遗憾）
`0.701%` 高于冻结解析规则的 `0.429%`，所以预注册四项门只通过三项，
composite success criterion（联合成功标准）未满足，机器字段保持
`structured_learnability_supported=false`。这不等于“没有可学习信号”：KNN 的
全任务 utility 与 mean-domain harm 更好，冻结解析规则的机会子集 regret 更低，
二者均不 Pareto-dominate（帕累托支配）对方。这回应了“只测两条手工规则”的
批评，但不把 learned selector 包装成成功算法。

匹配的 SIFt-RLS 和 DOS-ELM-style 结果现已完整进入证据集：SIFt 在全部任务上
近似中性，DOS-ELM-style 显著有害；在机会子集上二者只有约 `+0.11%` 的小幅
收益，明显低于解析规则和 learned selector。

## 信息隔离和选择结果

- meta-train：512 个任务；meta-validation：128；meta-test：256。seed 完全不重叠。
- selection lock 在任何 meta-test task 生成前写入，哈希
  `079b089c5695f7a0d8098b56fe472d35653adadfd6464d00e99d9224768ee335`。
- KNN `k` 从 `{1,3,5,9,17,33,65}` 中仅用 meta-validation 选择为 `9`。
- SIFt 域等效因子从冻结网格中仅用 meta-validation 选择为 `0.9`；逐样本因子
  为 `0.9**(1/96)`。
- 正式 manifest 记录 `git_dirty=false`、`testing=false`，且明确
  `fdst_reused=false`、`model_merging_in_scope=false`。
- 独立 validator 已从配置和全部 seeds 重新生成任务、重新选择超参数，并使
  `result.json` byte-for-byte（逐字节）一致。

## 主结果

所有数值均为相对 `f=1` 的 population balanced MSE utility（正值更好）；区间为
预注册 task-level bootstrap 95% interval。Regret 越低越好。

| Subset | Method | Utility % [95% CI] | Regret % | Factor MAE | Worst mean domain harm % |
|---|---|---:|---:|---:|---:|
| all 256 | fixed-f Oracle | +0.229 [0.168, 0.293] | 0.000 | 0.000 | 1.811 |
| all 256 | frozen `f*_pred` | -0.112 [-0.234, -0.005] | 0.341 | 0.081 | 2.184 |
| all 256 | learned KNN | **+0.073 [0.034, 0.117]** | **0.156** | **0.064** | 0.160 |
| all 256 | SIFt-RLS | +0.001 [-0.011, 0.013] | 0.228 | — | 0.813 |
| all 256 | DOS-ELM-style | -0.108 [-0.150, -0.069] | 0.337 | — | 2.186 |
| opportunity 40 | fixed-f Oracle | +1.221 [1.021, 1.450] | 0.000 | 0.000 | 5.660 |
| opportunity 40 | frozen `f*_pred` | **+0.791 [0.579, 1.019]** | **0.429** | **0.150** | 2.033 |
| opportunity 40 | learned KNN | +0.520 [0.330, 0.728] | 0.701 | 0.215 | -0.011 |
| opportunity 40 | SIFt-RLS | +0.109 [0.077, 0.143] | 1.112 | — | 0.528 |
| opportunity 40 | DOS-ELM-style | +0.118 [0.050, 0.202] | 1.103 | — | 1.307 |

## 冻结 composite success criterion（联合成功标准）

| Learned-selector gate | Result |
|---|---:|
| Factor MAE lower than constant `f=1` | PASS (`0.064 < 0.085`) |
| Opportunity-subset regret lower than frozen `f*_pred` | **FAIL** (`0.701 > 0.429`) |
| Opportunity-subset utility CI lower bound positive | PASS (`0.330% > 0`) |
| Worst mean domain harm <= 1% | PASS (`-0.011%`) |
| `structured_learnability_supported`（机器字段，不等于零信号） | **FALSE (3/4)** |

## Family-level interpretation（预指定 family，描述性）

- `recent_domains_more_reliable` 提供 40 个 opportunity task 中的 26 个。learned
  KNN 在该 opportunity 子集获得 `+0.788%`，但冻结解析规则为 `+1.027%`。
- `mapping_covariance_interaction` 有 8 个 opportunity task；learned KNN 保守选择
  `f=1`，utility 为 0，而解析规则为 `+0.390%`。
- `mixed_heterogeneity` 有 6 个 opportunity task；learned KNN 为 `+0.055%`。
- `recent_domains_less_reliable` 没有达到 0.5% 的 forgetting opportunity；这与
  “降低旧域权重无法补偿近期域更差可靠性”的方向一致，但不是独立理论证明。

因此 learned selector 学到了部分“近期域更可靠”的跨任务先验，但没有学到足以
覆盖 mapping–covariance scalar-grid opportunity 的通用映射。正结果不反驳 Proposition 3，
因为 meta-training 显式引入了跨任务分布假设；失败的联合门则阻止我们把原来的
两条规则失败重新解释为单纯的手工规则设计问题。

## SIFt / DOS-ELM 匹配范围

SIFt-RLS 使用相同 8 维线性头、4 域、每域 96 个样本、相同 ridge 和风险，但按
原算法做 384 次 rank-one information-subspace updates（秩一信息子空间更新），
保存 720 bytes 的 `R,W` 状态。DOS-ELM-style 使用 4 次域批次更新和 792 bytes 的
`R,C,W` 状态。它们匹配数据、头、信息权限和指标，但不匹配 FLOPs；论文必须
继续明确这个计算差异。

单独的 post-result compute audit（结果后计算审计）在固定的前 32 个 meta-test
任务上各重复 5 次，不改变任何结果或选择：本机 median per-task（每任务中位数）
为 `f=1` 域批次 0.016 ms、learned KNN 加最终求解 0.394 ms、SIFt-RLS
7.437 ms、DOS-ELM-style 0.053 ms。它只说明当前 8 维 NumPy 实现的计算差异，
不作为跨硬件速度主张。

本实验只属于 8 维合成线性头证据，不能作为自然 DINOv2 特征上
directional/anisotropic forgetting（方向性/各向异性遗忘）有效或无效的证明，
也不能把该结果冒充 crowd dual-head 的 headline comparison（主结果比较）。
