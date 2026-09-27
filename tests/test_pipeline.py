"""Pipeline + CLI tests (incl. offline fallback degradation path)."""

import json
import subprocess
import sys

from survforge.core.config import SurvConfig
from survforge.pipeline.pipeline import benchmark, format_table, run_split


def test_run_split_rows(split_data):
    tr, va, ho = split_data
    cfg = SurvConfig()
    out = run_split(tr, va, ho, cfg, seed=5)
    names = {r.backend for r in out["rows"]}
    assert "naive_random" in names and "numpy_cox" in names and "fusion_hpo" in names
    for r in out["rows"]:
        assert 0.0 <= r.c_index <= 1.0


def test_benchmark_structure_and_gates():
    cfg = SurvConfig(
        n_train=250,
        n_val=100,
        n_holdout=250,
        hpo_trials=6,
        hpo_timeout=8.0,
        seeds=(11, 22),
    )
    res = benchmark(cfg)
    assert set(res["table"]) >= {"naive_random", "numpy_cox", "fusion_hpo"}
    assert res["gates"]["flagship_vs_best_single"]["threshold"] == 0.05
    assert isinstance(res["gates"]["flagship_vs_best_single"]["pass"], bool)
    assert set(res["ablation"]) == {"fusion_uniform", "fusion_cweighted", "fusion_hpo"}
    assert len(res["failure_cases"]) == 3
    assert "fusion_hpo" in res["table"]


def test_failure_cases_structure():
    cfg = SurvConfig(
        n_train=250,
        n_val=100,
        n_holdout=250,
        hpo_trials=6,
        hpo_timeout=8.0,
        seeds=(11, 22),
    )
    res = benchmark(cfg)
    for fc in res["failure_cases"]:
        assert set(fc) >= {
            "true_risk_i",
            "pred_risk_i",
            "features_i",
            "rank_error_margin",
        }
        # genuine misranking: true order and predicted order disagree
        assert (fc["true_risk_i"] > fc["true_risk_j"]) != (
            fc["pred_risk_i"] > fc["pred_risk_j"]
        )


def test_format_table_contains_gates():
    cfg = SurvConfig(
        n_train=250,
        n_val=100,
        n_holdout=250,
        hpo_trials=6,
        hpo_timeout=8.0,
        seeds=(11, 22),
    )
    text = format_table(benchmark(cfg))
    assert "gate flagship" in text and "gate tier1" in text
    assert "PASS" in text or "FAIL" in text


def test_fallback_registry_without_optional_deps(monkeypatch, split_data):
    """Offline degradation: registry still yields >= 3 backends when both
    optional Tier-0 packages are 'missing'."""
    from survforge.survival import registry

    monkeypatch.setattr(registry, "lifelines_available", lambda: False)
    monkeypatch.setattr(registry, "xgb_available", lambda: False)
    tr, va, ho = split_data
    cfg = SurvConfig()
    factories = registry.build_single_backends(cfg)
    names = {f().name for f in factories}
    assert "numpy_cox" in names and "spline_cox" in names
    assert "lifelines_cox" not in names and "xgb_aft" not in names
    # fusion degrades to numpy-family members and still runs
    out = run_split(tr, va, ho, cfg, seed=5)
    names = {r.backend for r in out["rows"]}
    assert "fusion_hpo" in names


def test_cli_info_and_quick_demo(tmp_path):
    env_py = sys.executable
    r = subprocess.run(
        [env_py, "-m", "survforge.cli", "info"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0 and "numpy_cox" in r.stdout

    out = tmp_path / "bench.json"
    r = subprocess.run(
        [env_py, "-m", "survforge.cli", "demo", "--quick", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "table" in data and "gates" in data
