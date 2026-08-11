# RRCL → TMLR FDST C7-v3 单次正式结果审计

> 日期：2026-08-09（Asia/Shanghai）
> 性质：post-result audit（结果后审计），不修改预结果协议、不重跑实验
> 结论：`opportunity_absent=true`，`gap_confirmed=false`，`construction_candidate=false`

## 1. 单次运行与来源链

- 执行协议：`C7-fdst-v3`；
- split-freeze commit：`a398bc908c32670ac68f8e38b6fbb9ba404ccf3e`；
- 正式入口：`python3 scripts/run_fdst_c7.py`，无实验参数；
- `attempt.lock` 启动时间：`2026-08-09T08:51:44.257685+00:00`；
- 结果写入时间：`2026-08-09T09:28:06.801712+00:00`；
- 运行耗时：`1989.6202611923218` 秒；
- 正式结果：`runs_real/fdst_c7/fdst_c7.json`；
- 结果 SHA256：`0dfa51f7bf1008be170b43daf0969bb4c037f35c1cf50936d314770384bccdc3`；
- manifest SHA256：`e0eb6692b62b979579da1af97fb4e802c6127cd91fcdb40d9b914b6c2cb87c8c`；
- Appendix A SHA256：`3043263c071d24ee351f2a252c24147421c9554eb34830261de5c0747c8007a1`。

正式结果 `_provenance` 记录 `git_dirty=false`，commit 与冻结提交一致。结果目录和 feature cache（特征缓存）均被 `.gitignore` 排除；publication table（论文表）由独立验证器逐值绑定到正式 JSON。

## 2. 数据门与协议修订边界

正式 manifest 记录：

- 仅使用官方 `train_data`；官方 `test_data` 未进入选择、划分、拟合、验证或评估；
- 全量检查 9,000 帧、232,244 个标注点，空帧为 0；
- 319 帧包含 327 个需裁剪的 near-boundary points（近边界点）；
- 最大横向/纵向越界分别为 8/4 像素；
- 超出冻结每轴 1% 容忍带的点为 0；
- 60 个训练视频覆盖 13 个 reviewed scene identities（复核场景身份）；
- 正式结果使用六个不同场景，每场景一个 150 帧视频，按 10 帧块冻结为 90/30/30 fit/validation/test。

预结果 v3 文档中的 134 点是 v3 修订前置的非正式诊断计数；正式全量 preflight 在任何模型访问和 `attempt.lock` 之前记录 327 点，并将该统计冻结进 manifest。两者均保留，不用结果后修改协议文档来抹平差异；正式门禁与论文报告以冻结 manifest 的全量统计为准。

## 3. 正式结果

| 方法 | balanced rel-MAE ↓ | 相对 `f=1` 改善 ↑ | positive | safe |
|---|---:|---:|---|---|
| `f=1` | 0.0944616464 | 0.000% | — | — |
| fixed-`f` Oracle（诊断） | 0.0944616464 | 0.000% | false | — |
| `fstar_pred` | 0.0944616464 | 0.000% | false | true |
| `shrinkage_gamma` | 0.0944616464 | 约 0.000% | false | true |
| `precision_weighting` | 0.1152964125 | -22.056% | false | false |

Oracle 的最优固定因子为 `f=1`。2,000 次 selection-adjusted paired block bootstrap（选择校正配对块自助法）的改善均值和 95% 区间均为 0。因此先于任何 selector（选择器）恢复率问题，`0.5%` opportunity gate（机会门）已经失败。

## 4. 允许与禁止的论文表述

允许：

> Under the frozen C7-fdst-v3 setting, one previously unseen FDST source shows no material fixed-factor opportunity: the diagnostic Oracle selects `f=1`, neither prespecified train-only rule improves it, and precision weighting is harmful.

禁止：

- “所有模型合并或历史加权都无效”；
- “六个 FDST 场景是六次独立复现”；
- “FDST 反驳了开发数据上的 opportunity--selection gap”；
- “没有 Oracle opportunity 证明任何 train-only selector 都不可能成功”；
- 使用 FDST 结果继续调 selector、阈值、因子网格或数据顺序。

## 5. 独立验收入口

```bash
python3 scripts/validate_fdst_c7_result.py
```

该命令只读检查结果 SHA256、single-shot lock（单次锁）、冻结提交、manifest/Appendix 哈希、数据审计、六场景结构、Oracle 结果、最终分类和论文 CSV；它不会加载模型、生成特征或重跑实验。
