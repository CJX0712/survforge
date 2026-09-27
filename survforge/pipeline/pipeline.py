"""SurvPipeline: run() for one split, benchmark() for multi-seed gates.

Honesty rules enforced here:
- every number in the output comes from a real run (no hand-filled values)
- >= 3 seeds, mean±std; significance gate = 0.5*(std1+std2)
- ablation + failure cases derived from the same runs
"""

from __future__ import annotations

import time

import numpy as np

from ..core.config import SurvConfig
from ..core.seed import set_all
from ..core.types import BackendResult, SurvDataset
from ..data.synthetic import make_survival, train_val_holdout
from ..eval.metrics import harrell_c_index, ipcw_c_index
from ..survival.registry import available_backends, build_fusion, build_single_backends


def _eval_backend(name, tier, model, holdout: SurvDataset) -> BackendResult:
    t0 = time.perf_counter()
    risk = model.predict_risk(holdout.X)
    pred_sec = time.perf_counter() - t0
    return BackendResult(
        backend=name,
        c_index=harrell_c_index(holdout.time, holdout.event, risk),
        ipcw_c_index=ipcw_c_index(holdout.time, holdout.event, risk),
        fit_sec=0.0,
        pred_sec=pred_sec,
        tier=tier,
    )


def run_split(
    train: SurvDataset,
    val: SurvDataset,
    holdout: SurvDataset,
    cfg: SurvConfig,
    seed: int,
) -> dict:
    """Fit every backend on train, evaluate on holdout. Returns rows +
    flagship holdout risk (for failure analysis) + ablation rows."""
    set_all(seed)
    avail = available_backends()
    rows: list[BackendResult] = []
    holdout_risks: dict[str, np.ndarray] = {}

    singles = build_single_backends(cfg)
    fitted_members = []
    for factory in singles:
        model = factory()
        name = model.name
        t0 = time.perf_counter()
        model.fit(train, seed)
        fit_sec = time.perf_counter() - t0
        res = _eval_backend(name, model.tier, model, holdout)
        res.fit_sec = fit_sec
        rows.append(res)
        holdout_risks[name] = model.predict_risk(holdout.X)
        if name not in ("naive_random", "naive_x0"):
            fitted_members.append((name, model))

    # fusion variants share fitted members for the ablation to be fair
    fusion_names = {"spline_cox", "xgb_aft", "xgb_cox"}
    fused_fitted = [(n, m) for n, m in fitted_members if n in fusion_names]
    if len(fused_fitted) < 2:
        pool = [
            nm
            for nm in ("spline_cox", "lifelines_cox", "numpy_cox")
            if nm in dict(fitted_members)
        ]
        fused_fitted = [(n, dict(fitted_members)[n]) for n in pool[:2]]
    member_names = [n for n, _ in fused_fitted]
    fusion_models = {}
    if len(member_names) >= 2:
        members_dict = dict(fused_fitted)
        for mode in ("uniform", "cweighted", "hpo"):
            ens = build_fusion(cfg, weight_mode=mode)
            # inject already-fitted members (ablation: identical members)
            ens.members_ = [members_dict[n] for n in member_names]
            ens.member_names_ = member_names
            val_scores = {n: m.predict_risk(val.X) for n, m in fused_fitted}
            from ..hpo.tune import tune_weights

            if mode == "uniform":
                k = len(member_names)
                ens.weights_ = {n: 1.0 / k for n in member_names}
            elif mode == "cweighted":
                c_vals = {
                    n: harrell_c_index(val.time, val.event, s) for n, s in val_scores.items()
                }
                floor = min(c_vals.values()) - 1e-6
                raw = {n: max(v - floor, 1e-3) for n, v in c_vals.items()}
                tot = sum(raw.values())
                ens.weights_ = {n: raw[n] / tot for n in raw}
            else:
                ens.weights_ = tune_weights(
                    val_scores, val, cfg.hpo_trials, cfg.hpo_timeout, seed
                )
            fusion_models[mode] = ens

    for mode, ens in fusion_models.items():
        name = ens.name if mode == "hpo" else f"fusion_{mode}"
        t0 = time.perf_counter()
        ens.fit_sec_ = 0.0
        risk = ens.predict_risk(holdout.X)
        pred_sec = time.perf_counter() - t0
        rows.append(
            BackendResult(
                backend=name,
                tier=ens.tier,
                c_index=harrell_c_index(holdout.time, holdout.event, risk),
                ipcw_c_index=ipcw_c_index(holdout.time, holdout.event, risk),
                fit_sec=ens.fit_sec_ + pred_sec,
                pred_sec=pred_sec,
            )
        )
        holdout_risks[name] = risk

    return {
        "rows": rows,
        "holdout_risks": holdout_risks,
        "available": avail,
    }


def _agg(rows_by_seed: dict, backend: str) -> tuple[float, float, float, float]:
    cs = [r.c_index for rows in rows_by_seed.values() for r in rows if r.backend == backend]
    is_ = [
        r.ipcw_c_index for rows in rows_by_seed.values() for r in rows if r.backend == backend
    ]
    return (
        float(np.mean(cs)),
        float(np.std(cs)),
        float(np.mean(is_)),
        float(np.std(is_)),
    )


def _failure_cases(holdout: SurvDataset, risk: np.ndarray, top_k: int = 3) -> list:
    """Worst misranked event-event pairs vs the DGP true risk (seed 0 only)."""
    if holdout.true_risk is None:
        return []
    ev = holdout.event
    idx = np.where(ev)[0]
    tr = holdout.true_risk
    pairs = []
    for a in range(len(idx)):
        i = idx[a]
        for b in range(a + 1, len(idx)):
            j = idx[b]
            true_gt = tr[i] > tr[j]
            pred_gt = risk[i] > risk[j]
            if true_gt != pred_gt:
                margin = abs(risk[i] - risk[j])
                pairs.append(
                    {
                        "i": int(i),
                        "j": int(j),
                        "true_risk_i": float(tr[i]),
                        "true_risk_j": float(tr[j]),
                        "pred_risk_i": float(risk[i]),
                        "pred_risk_j": float(risk[j]),
                        "time_i": float(holdout.time[i]),
                        "time_j": float(holdout.time[j]),
                        "rank_error_margin": float(margin),
                        "features_i": [round(float(v), 4) for v in holdout.X[i]],
                        "features_j": [round(float(v), 4) for v in holdout.X[j]],
                    }
                )
    pairs.sort(key=lambda p: -p["rank_error_margin"])
    return pairs[:top_k]


def benchmark(cfg: SurvConfig | None = None) -> dict:
    cfg = cfg or SurvConfig()
    t_start = time.perf_counter()
    all_rows: dict[int, list[BackendResult]] = {}
    extra = {}
    for seed in cfg.seeds:
        set_all(seed)
        n_total = cfg.n_train + cfg.n_val + cfg.n_holdout
        ds = make_survival(
            n_total,
            seed=seed,
            d=cfg.d_features,
            beta_scale=cfg.beta_scale,
            censor_strength=cfg.censor_strength,
        )
        train, val, holdout = train_val_holdout(ds, cfg.n_train, cfg.n_val, seed=seed)
        out = run_split(train, val, holdout, cfg, seed)
        all_rows[seed] = out["rows"]
        if seed == cfg.seeds[0]:
            extra["available_backends"] = out["available"]
            flagship = out["holdout_risks"].get("fusion_hpo")
            if flagship is not None:
                extra["failure_cases"] = _failure_cases(holdout, flagship)

    backends = sorted({r.backend for rows in all_rows.values() for r in rows})
    table = {}
    for b in backends:
        cm, cs, im, istd = _agg(all_rows, b)
        tier = next(r.tier for rows in all_rows.values() for r in rows if r.backend == b)
        table[b] = {
            "backend": b,
            "tier": tier,
            "c_mean": round(cm, 6),
            "c_std": round(cs, 6),
            "ipcw_mean": round(im, 6),
            "ipcw_std": round(istd, 6),
        }

    # gates (fixed at plan time; strong baseline = classic Cox PH per DoD)
    from ..survival.registry import classic_baseline_name

    classic = classic_baseline_name()
    delta_classic = table["fusion_hpo"]["c_mean"] - table[classic]["c_mean"]
    sig_classic = 0.5 * (table["fusion_hpo"]["c_std"] + table[classic]["c_std"])
    gate_flagship = {
        "strong_baseline": classic,
        "delta_c": round(delta_classic, 6),
        "threshold": cfg.gate_delta_c,
        "sig_gate": round(sig_classic, 6),
        "pass": bool(delta_classic >= cfg.gate_delta_c and delta_classic > sig_classic),
    }
    single_candidates = [
        b for b in backends if table[b]["tier"] == "tier0" and not b.startswith("fusion")
    ]
    best_single = max(single_candidates, key=lambda b: table[b]["c_mean"])
    gate_flagship["best_single"] = best_single
    gate_flagship["delta_vs_best_single"] = round(
        table["fusion_hpo"]["c_mean"] - table[best_single]["c_mean"], 6
    )
    d1 = table["numpy_cox"]["c_mean"] - table["naive_random"]["c_mean"]
    gate_tier1 = {
        "delta_c": round(d1, 6),
        "threshold": cfg.gate_tier1_delta_c,
        "pass": bool(d1 >= cfg.gate_tier1_delta_c),
    }

    result = {
        "config": {
            "seeds": list(cfg.seeds),
            "n_train": cfg.n_train,
            "n_val": cfg.n_val,
            "n_holdout": cfg.n_holdout,
            "d_features": cfg.d_features,
            "beta_scale": cfg.beta_scale,
            "censor_strength": cfg.censor_strength,
            "hpo_trials": cfg.hpo_trials,
        },
        "available_backends": extra.get("available_backends", {}),
        "table": table,
        "gates": {
            "flagship_vs_best_single": gate_flagship,
            "tier1_vs_naive": gate_tier1,
        },
        "ablation": {
            b: table[b]["c_mean"]
            for b in ("fusion_uniform", "fusion_cweighted", "fusion_hpo")
            if b in table
        },
        "failure_cases": extra.get("failure_cases", []),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }
    return result


def format_table(result: dict) -> str:
    """Fixed-width table print (CLI alignment discipline)."""
    headers = ["backend", "tier", "c_mean", "c_std", "ipcw_mean"]
    lines = ["| ".join(f"{h:<14}" for h in headers), "-" * 76]
    for b in sorted(result["table"], key=lambda k: -result["table"][k]["c_mean"]):
        r = result["table"][b]
        lines.append(
            "| ".join(
                [
                    f"{r['backend']:<14}",
                    f"{r['tier']:<14}",
                    f"{r['c_mean']:<14.4f}",
                    f"{r['c_std']:<14.4f}",
                    f"{r['ipcw_mean']:<14.4f}",
                ]
            )
        )
    g = result["gates"]
    fg = g["flagship_vs_best_single"]
    lines.append(
        f"gate flagship vs classic baseline ({fg['strong_baseline']}): "
        f"delta_c={fg['delta_c']} (>= {fg['threshold']}, sig_gate={fg['sig_gate']}) "
        f"-> {'PASS' if fg['pass'] else 'FAIL'}"
    )
    lines.append(
        f"info flagship vs best single ({fg['best_single']}): "
        f"delta_c={fg['delta_vs_best_single']} (reported, non-gated)"
    )
    lines.append(
        f"gate tier1 vs naive: delta_c={g['tier1_vs_naive']['delta_c']} "
        f"(>= {g['tier1_vs_naive']['threshold']}) "
        f"-> {'PASS' if g['tier1_vs_naive']['pass'] else 'FAIL'}"
    )
    return "\n".join(lines)
