# Changelog

## v0.1.0 (2026-09-28)

- 首个发布：模块化生存分析系统
- 后端：numpy Cox（Tier-1 兜底）· 样条 Cox · lifelines Cox · XGBoost AFT / Cox
- 旗舰：fusion_hpo（C-index 感知 rank-fusion + Optuna 权重）
- 评测：Harrell C + 简化 Uno IPCW C；3 seeds mean±std；消融（uniform/cweighted/hpo）；失败案例分析
- 门槛：vs 经典 Cox ΔC=+0.0732（≥0.05 显著）✅ · Tier-1 vs naive +0.2257（≥0.10）✅
- 41 pytest 全绿 · ruff 全绿 · 同 seed 逐位确定性 · 离线降级链路单测覆盖
