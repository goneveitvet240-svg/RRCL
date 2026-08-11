# RRCL TMLR controlled synthetic study（受控合成研究）预注册协议 v1.0

> 冻结候选日期：2026-08-07（Asia/Shanghai）
> 角色：解释 opportunity--selection gap（机会--选择鸿沟）的机制边界，不用于开发新 selector（选择器）
> 当前状态：**唯一正式运行已于 freeze commit `46a015c...` 完成**。320 个 scenario 全部写入，独立 validator 复算通过；结果见 `docs/RRCL_TMLR_SYNTHETIC_RESULTS_2026-08-07.md`。不得修改协议后重跑。

## 1. 研究问题与禁止事项

本研究只回答四个受控因素如何改变 balanced final risk（平衡最终风险）以及 train-only rules（仅训练侧规则）能否选择有用权重：

1. label scale（标签尺度）；
2. domain heteroscedasticity / reliability（域级异方差 / 可靠性）；
3. linear mapping change（线性映射变化）；
4. anisotropic feature covariance（各向异性特征协方差）。

禁止根据正式结果修改因子幅度、seed（随机种子）、factor grid（因子网格）、方法参数、主指标或判定门。任何修改均升级为 v2，并将 v1 的全部正式结果保留为历史记录。合成结果不得写成自然域复现，也不得单独解锁“真实 crowd 数据中的收益来自 mapping drift”这一主张。

## 2. 数据生成模型

共有 `T=4` 个顺序域、特征维度 `d=8`。每个域独立生成 `n_train=512`、`n_test=4096` 个样本：

\[
x_{ki}\sim\mathcal N(0,\Sigma_k),\qquad
y_{ki}=s_k\left(x_{ki}^{\top}\beta_k+\epsilon_{ki}\right),\qquad
\epsilon_{ki}\sim\mathcal N(0,\sigma_k^2).
\]

训练和测试来自同一个明示的域分布；本协议不引入隐藏 train--test shift（训练--测试偏移）。部署模型是单个 shared linear head（共享线性头），使用 ridge objective（岭目标）`lambda=1.0`。主分析先除以已知域尺度 `s_k`，与论文的 task-aware target normalization（任务可知目标归一化）一致；raw-scale analysis（原尺度分析）只用于标签尺度对照。

基础向量固定为

\[
\beta_0=(1,-1,0.75,-0.75,0.5,-0.5,0.25,-0.25)/\sqrt{3.75}.
\]

mapping-change 方向固定为与 `beta_0` 正交并单位化的

\[
u=(1,1,-1,-1,1,1,-1,-1)/\sqrt 8,
\]

域系数 `a=(-0.75,-0.25,0.25,0.75)`，启用 mapping change 时 `beta_k=beta_0+0.8 a_k u`，否则 `beta_k=beta_0`。

## 3. 四因素全因子设计

执行完整 `2^4=16` cells（条件格），不得删除“不好看”的 cell：

| 因素 | `0` 水平 | `1` 水平 |
|---|---|---|
| label scale | `s=(1,1,1,1)` | `s=(0.5,1,2,4)` |
| reliability | `sigma=(1,1,1,1)` | `sigma=(0.25,0.5,1,2)` |
| mapping change | `beta_k=beta_0` | 使用第 2 节固定 `beta_k` |
| covariance anisotropy | `Sigma_k=I` | `Sigma_k=Q diag(v_k) Q^T` |

`Q` 是固定 Hadamard orthogonal matrix（Hadamard 正交矩阵）；四个 `v_k` 依次循环
`(4,4,1,1,0.25,0.25,1,1)` 的坐标，使域间信息方向不同，同时保持相同 trace（迹）。不得根据运行结果替换 `Q` 或谱。

正式 seeds 固定为
`[2026080701, 2026080702, 2026080703, 2026080704, 2026080705, 2026080706, 2026080707, 2026080708, 2026080709, 2026080710]`。
每个 cell 同时运行主顺序 `(0,1,2,3)` 与完全反序 `(3,2,1,0)`；反序是 directionality check（方向性检查），不是额外独立样本。

## 4. 冻结方法与信息权限

所有方法共享相同训练样本、域边界、尺度和 ridge 参数：

| ID | 方法 | 可用信息 | 角色 |
|---|---|---|---|
| M0 | `f=1` | 全部训练充分统计量 | joint-equivalent reference（联合等价参考） |
| M1 | fixed-`f` Oracle，`f in {0.00,0.05,...,1.00}` | 测试风险 | diagnostic only（仅诊断），不可部署 |
| M2 | analytic `f*_pred` | 完整训练样本与训练侧统计 | 冻结失败规则之一 |
| M3 | precision weighting | 训练域内 fit/validation 残差 | 冻结 reliability baseline（可靠性基线） |
| M4 | shrinkage-gamma per-boundary proxy | 训练侧保留块 | 冻结 C4 规则；门失败不等于从方法表删除 |
| M5 | VFF-RLS | 仅过去和当前训练流 | 参数必须在首次正式运行前由代码常量固定 |

SIFt/DOS-ELM 只有在 runner freeze commit（运行器冻结提交）前完成相同线性头、相同域边界、相同信息权限和相同计算预算的机械适配时才进入 secondary table（次表）；否则记录 `not computationally matched`，不得在看过结果后补入或删除。

## 5. 主指标与解析参照

主指标是四域均匀混合分布上的 population normalized MSE（总体归一化 MSE）。它由高精度解析矩阵表达式计算，不使用有限 test set 选 Oracle。有限样本 `n_test` 只生成 paired empirical check（配对经验检查）。

每个 cell/seed/order 必须输出：

- `R_f1`、`R_oracle`、`R_method`；
- Oracle opportunity：`(R_f1-R_oracle)/R_f1`；
- deployable utility：`(R_f1-R_method)/R_f1`；
- Oracle regret：`(R_method-R_oracle)/R_f1`；
- `f_oracle`、`f_pred`、factor absolute error；
- 最终逐域 MSE 和最大单域 harm（伤害）；
- 数据配置、seed、顺序、代码 commit、dirty flag 和完整 manifest hash。

所有 factor ties（并列）选择最大的 `f`，即最保守、最接近 `f=1` 的因子。不得用 rel-MAE 选因子后用 MSE 解释成功。

## 6. 预注册机制检查

这些检查是 falsification checks（证伪检查），不是保证得到正结果的成功门：

1. **Null**：四因素全为 `0` 时，population Oracle opportunity 应不超过 `0.1%`；超过则优先判定实现、ridge 缩放或风险公式错误。
2. **Scale isolation**：只开启 label scale 时，raw-scale 风险可变化，但已知 `s_k` 归一化后的 population curve 应与 Null 在 `1e-8` 数值容差内一致。
3. **Reliability isolation**：只开启 reliability 时，precision weighting 的方向应给予低噪声域不小于高噪声域的权重；是否优于 `f=1` 由实际风险报告，不把方向正确替代部署收益。
4. **Mapping isolation**：只开启 mapping change、保持均匀 balanced risk 与共同 `Sigma=I` 时，mapping change 本身不预注册为 forgetting opportunity；若 `f=1` 仍最优，这是有信息量的负控制。
5. **Anisotropy interaction**：mapping change 与 covariance anisotropy 的交互是否产生非均匀权重机会由解析 population curve 判定；结果为零同样保留，不调谱追求正结果。

只有合成数据能把某因素开/关时，正文才可写“under this controlled model（在该受控模型下）该因素改变了机会或选择”；不得据此唯一归因真实数据。

## 7. 不确定性与判定

seed 是独立生成重复，order 不是。对每个 cell，先在同一 seed 内平均两个 order，再对 10 个 seed 的 paired difference（配对差）做 `B=10000` percentile bootstrap（百分位自助法），报告 95% interval（区间）。同时报告全部 seed，不做异常值删除。

- `opportunity present`：平均 Oracle opportunity `>=0.5%` 且 95% interval 下界 `>0`；
- `selector useful`：相对 `f=1` 的平均 deployable utility `>=0.5%`、95% interval 下界 `>0`，且任何域的平均相对 harm 不超过 `1%`；
- `opportunity--selection gap`：`opportunity present=true` 且该 train-only rule 的 `selector useful=false`。

这些标签只用于结构化汇总；论文必须同时给出连续效应量，不得只报 PASS/FAIL。

## 8. 运行与产物门

正式 runner 必须：

1. 无实验参数 CLI；只允许 `--preflight` 和明确写入临时目录的 `--testing`；
2. 在任何正式计算前验证 clean worktree、配置 SHA256 与不存在历史正式结果；
3. 以 `O_CREAT|O_EXCL` 创建永久 attempt lock；崩溃后不得自动重跑；
4. 每个 cell/seed/order 写一份 raw JSON，最后机械生成 summary JSON、CSV 和图；
5. validator 从 raw JSON 重算全部主表、主图、interval 和判定；
6. 不覆盖旧产物，正式路径固定为 `runs_real/tmlr_synthetic_v1/`。

实现对应关系：

- 配置：`configs/tmlr_synthetic_v1.json`，SHA256 `7b9a9164ba827f0fefabde47162986a08c22f2a281d78e18ba52fa7b3489ad75`；
- runner：`scripts/run_tmlr_synthetic_v1.py`；
- summarizer：`scripts/summarize_tmlr_synthetic_v1.py`；
- validator：`scripts/validate_tmlr_synthetic_v1.py`；
- 回归测试：`tests/test_tmlr_synthetic_v1.py`。

执行命令冻结为：

```bash
# 只检查是否满足正式运行条件，不计算任何 scenario
python3 scripts/run_tmlr_synthetic_v1.py --preflight

# 唯一正式运行入口：320 scenarios，自动生成 summary JSON/CSV
python3 scripts/run_tmlr_synthetic_v1.py

# 正式运行完成后的独立复算验收
python3 scripts/validate_tmlr_synthetic_v1.py
```

正式运行来自 clean freeze commit
`46a015c1af52c24cd722cfb5998df61596a2e161`；manifest 记录
`git_dirty=false`、`testing=false`。正式输出目录和永久 attempt lock 已存在，
runner 会拒绝再次运行。后续只允许从 raw JSON 机械重算、制表、绘图和文字解释。
