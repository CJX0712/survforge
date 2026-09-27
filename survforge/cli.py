"""CLI entry: python -m survforge.cli {demo|benchmark|info}"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import SurvConfig


def _utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main(argv=None) -> int:
    _utf8()
    parser = argparse.ArgumentParser(prog="survforge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="end-to-end demo, writes benchmark.json")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--out", default="benchmark.json")
    p_demo.add_argument("--quick", action="store_true", help="CI smoke: small + fast")

    p_bench = sub.add_parser("benchmark", help="full multi-seed benchmark")
    p_bench.add_argument("--out", default="benchmark.json")

    sub.add_parser("info", help="show backend availability")

    args = parser.parse_args(argv)

    if args.cmd == "info":
        from .survival.registry import available_backends

        print(json.dumps(available_backends(), indent=2))
        return 0

    if args.cmd == "demo":
        cfg = SurvConfig()
        if args.quick:
            cfg.n_train, cfg.n_val, cfg.n_holdout = 400, 150, 300
            cfg.hpo_trials, cfg.hpo_timeout = 12, 10.0
            cfg.rsf_n_estimators = cfg.gbs_n_estimators = 120
    else:
        cfg = SurvConfig()

    from .pipeline.pipeline import benchmark, format_table

    result = benchmark(cfg)
    print(format_table(result))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"benchmark.json written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
