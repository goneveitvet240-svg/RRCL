# RRCL TMLR selector-v2 预结果协议

> 冻结候选日期：2026-08-10（Asia/Shanghai）<br>
> 状态：**PRE-RESULT / NOT YET FORMALLY RUN（预结果、尚未正式运行）**<br>
> 目的：只回应两项审稿风险——“两条手工规则不能代表可学习选择器”和“缺少
> SIFt-RLS / DOS-ELM 匹配基线”。本协议不涉及 model merging（模型合并），
> 不修改 synthetic-v1，不重新使用 FDST。

## 1. 问题和证据边界

实验在明示的合成 task distribution（任务分布）中测试：当 meta-training
（元训练）阶段允许使用模拟器给出的 population Oracle curves（总体 Oracle
曲线），部署到一个完全留出的新任务时，能否只从该任务的训练侧统计选择固定
遗忘因子。正结果只说明“在这个结构化任务分布中，规则设计而非信息本身可能是
瓶颈”；负结果也不能推出所有 train-only selectors（仅训练侧选择器）不可能。

SIFt-RLS 和 DOS-ELM-style 与其他方法共享相同的线性头、训练样本、域顺序、
ridge、目标归一化和最终 population balanced MSE（总体平衡 MSE）。SIFt-RLS
按原算法逐样本更新，因此计算量高于域批次充分统计更新；报告运行时间和状态
大小，不伪称 FLOP 完全相同。

## 2. 严格留出结构

三组 scenario seed（场景种子）互不相交：

| Split | Seeds | 用途 |
|---|---:|---|
| meta-train | `202608100000..202608100511` | KNN selector 的训练任务库和 Oracle 曲线 |
| meta-validation | `202608110000..202608110127` | 只选择 KNN 的 `k` 和 SIFt 的固定域等效因子 |
| meta-test | `202608120000..202608120255` | 唯一主结果；超参数锁写入后才允许生成 |

runner 必须在生成任何 meta-test task 之前，以 `O_EXCL` 写入
`selection.lock.json`，记录 scaler、KNN `k`、SIFt 因子、meta-train/meta-validation
记录哈希和代码 commit。测试集不得用于选择特征、距离、网格或阈值。

## 3. 生成分布

每个任务有 4 个顺序域、8 维特征、每域 96 个训练样本，训练和 population
test risk 来自同一高斯线性域。部署风险始终对四域均匀加权；不利用未知
deployment weights（部署权重）制造优势。

生成器固定四个等概率 family：近期域更可靠、近期域更不可靠、mapping–covariance
interaction（映射–协方差交互）、mixed heterogeneity（混合异质性）。mapping
amplitude、noise base/ratio 和 covariance spectrum 的范围固定在配置文件中。
family 由 `seed % 4` 决定。所有随机正交方向、协方差和训练样本均由 scenario
seed 确定。

这些范围在正式结果前冻结，目的是覆盖既有 v1 没覆盖的更强异质性，而不是保证
learned selector 得到正结果。全部 meta-test 任务必须保留，包括没有 Oracle
opportunity 的任务。

## 4. 方法

### M0：`f=1`

联合等价参考。

### M1：fixed-f Oracle

在配置中的 factor grid 上用 population balanced MSE 选因子；只作诊断和
meta-training label，不可部署。并列时选最大因子。

### M2：冻结解析 `f*_pred`

沿用论文现有 train-only approximation（仅训练侧近似），不调公式。

### M3：learned KNN curve selector

每个任务从训练数据提取有序域级和成对 invariant summaries（不变量摘要）：局部
ridge 解的范数、训练保留块残差、目标方差、样本协方差 trace/condition、局部解
差异、cosine 和 covariance-weighted disagreement。标准化均值和标准差只从
meta-train 计算。

对一个新任务寻找 `k` 个最近 meta-train 任务，平均它们按各自 `f=1` 风险归一化
的 Oracle 曲线，然后选平均风险最小的因子；并列时选最大因子。`k` 只在
meta-validation 的平均 normalized regret（归一化遗憾）上从冻结网格选择；并列
时选择较大的 `k`。

### M4：SIFt-RLS

使用已核对的原始 information-subspace update（信息子空间更新）。为避免把域级
因子在 96 个样本上重复应用，候选域等效因子 `f_domain` 被机械换算为
`f_sample=f_domain**(1/96)`。候选 `f_domain` 只在 meta-validation 的平均
population balanced MSE 上选择；meta-test 使用一个冻结常数，不查看其标签或
风险。更新顺序是生成器给出的原始训练顺序，不重排。

### M5：DOS-ELM-style regression adaptation

使用相同冻结特征线性头。按原文设计矩阵缩放解释，历史充分统计量乘
`lambda**2`。当前域的 train normalized RMSE（训练归一化 RMSE）只更新下一域
的因子，因此信息流是因果的。明确称 style adaptation（风格适配），不称原论文
回归协议的精确复现。

## 5. 主指标和判读

每个 meta-test task 报告：Oracle opportunity、factor absolute error、相对
`f=1` utility、相对 Oracle regret、逐域 harm。主要分析同时报告所有 256 个任务
和 opportunity `>=0.5%` 的预指定子集。

learned selector 的预注册判读：

1. factor MAE 小于 constant `f=1`；
2. opportunity 子集上的 mean regret 小于冻结 `f*_pred`；
3. opportunity 子集 mean utility 的 95% bootstrap interval 下界大于 0；
4. 最大 mean per-domain harm 不超过 1%。

四项全满足才标记为 `structured_learnability_supported=true`。任何一项失败均保留，
不得改 KNN 特征、距离、`k` 网格或生成范围后覆盖 v2。

SIFt-RLS 和 DOS-ELM 不设“必须胜出”的成功门；回应审稿批评所需的是同协议结果
完整进入表格。若其效用为负，同样报告；若为正，也不能推广为自然数据优势。

## 6. 正式运行门

1. 正式 runner 无实验超参数 CLI，只允许 `--preflight` 和写入非正式临时路径的
   `--testing`。
2. 正式运行要求 clean git worktree、固定配置路径、固定输出路径且输出不存在。
3. 先创建永久 attempt lock；meta-validation 完成后创建不可覆盖的 selection lock；
   selection lock 存在后才生成 meta-test。
4. 唯一正式运行后不得重跑或修改 v2；任何实质修改升级 v3。
5. validator 从 config 和 seeds 重新生成全部任务、重新选择超参数并复算结果。

冻结命令将是：

```bash
python3 scripts/run_tmlr_selector_v2.py --preflight
python3 scripts/run_tmlr_selector_v2.py
python3 scripts/validate_tmlr_selector_v2.py
```
