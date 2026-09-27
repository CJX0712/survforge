# SurvForge

**模块化生存分析（Survival Analysis）系统** — Cox PH / 样条 Cox / XGBoost AFT / XGBoost Cox + C-index 感知 rank-fusion 旗舰，纯 numpy 离线兜底，零下载 CPU 可跑。

![CI](https://github.com/CJX0712/survforge/actions/workflows/ci.yml/badge.svg)
![Release](https://img.shields.io/github/v/release/CJX0712/survforge)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![质量等级](https://img.shields.io/badge/质量等级-S-brightgreen)

作者：**晨星** · 仓库：`CJX0712/survforge` · v0.1.0

## 一键复现

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m survforge.cli demo --out benchmark.json   # ~6s, CPU
.venv/Scripts/python -m pytest -q                                  # 41 tests
```

Docker: `docker build -t survforge . && docker run survforge`

## 核心结果（3 seeds mean±std，benchmark.json 为真实运行输出）

| 后端 | 层级 | Harrell C | IPCW C |
|------|------|-----------|--------|
| **fusion_hpo（旗舰）** | tier0 | **0.7855 ± 0.0086** | 0.7772 |
| fusion_cweighted（消融） | tier0 | 0.7865 ± 0.0041 | 0.7786 |
| fusion_uniform（消融） | tier0 | 0.7849 ± 0.0114 | 0.7768 |
| xgb_aft | tier0 | 0.7852 ± 0.0031 | 0.7769 |
| xgb_cox | tier0 | 0.7823 ± 0.0039 | 0.7743 |
| spline_cox | tier1 | 0.7415 ± 0.0278 | 0.7321 |
| lifelines_cox（经典强基线） | tier0 | 0.7124 ± 0.0031 | 0.7010 |
| numpy_cox（离线兜底） | tier1 | 0.7122 ± 0.0031 | 0.7008 |
| naive_x0 | baseline | 0.6399 ± 0.0057 | 0.6334 |
| naive_random | baseline | 0.4865 ± 0.0143 | 0.4844 |

**门槛（方案阶段定死，DoD）：**
- 旗舰 vs 经典强基线（Cox PH）：ΔC = **+0.0732** ≥ 0.05，显著性门槛 ½(std₁+std₂)=0.0059 ✅
- Tier-1 离线兜底 vs naive_random：ΔC = **+0.2257** ≥ 0.10 ✅
- 旗舰 vs 最强单模型（xgb_aft）：+0.0004（非劣，诚实报告，不设门槛）

**消融**：uniform 0.7849 → cweighted 0.7865 → hpo 0.7855（权重选择对融合敏感度低；融合价值来自成员误差异构本身）。
**失败案例**：3 条典型误排事件对（true_risk 与预测排序不一致），见 `benchmark.json`。

## 架构

```
cli → pipeline → {data, hpo, survival, eval} → core
survival/: numpy_cox(T1) · spline_cox(T1) · lifelines_cox(T0) · xgb_aft(T0) · xgb_cox(T0)
           ensemble(FusionEnsemble: rank-fusion + Optuna 权重) · registry(可用性探测/降级)
```

- **统一契约**：所有后端 `fit(train, seed)` → `predict_risk(X)`，分数越大风险越高。
- **确定性**：唯一 seed 入口 `core.seed.set_all`；同 seed 两次运行核心指标逐位一致（含 failure_cases）。
- **离线降级**：lifelines/xgboost 缺失时 registry 自动裁剪成员，numpy 兜底链路有单测覆盖。

## 文档

- [架构说明](docs/architecture.md)（选型依据 / SOTA 对标 / 门槛定义 / 踩坑记录）
- [模型卡](docs/model_card.md)（用途 / 数据 / 指标 / 局限）

## License

MIT © 晨星 (CJX0712)
