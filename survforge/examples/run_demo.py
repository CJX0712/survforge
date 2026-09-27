"""End-to-end demo: benchmark -> benchmark.json (deterministic, seed=42)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", "--seed", "42", "--out", "benchmark.json"]))
