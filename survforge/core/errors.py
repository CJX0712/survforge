"""Stable error codes E100~E500."""

from __future__ import annotations


class SurvForgeError(Exception):
    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class E100ConfigError(SurvForgeError):
    code = "E100"


class E200DataError(SurvForgeError):
    code = "E200"


class E300BackendError(SurvForgeError):
    code = "E300"


class E400EvalError(SurvForgeError):
    code = "E400"


class E500PipelineError(SurvForgeError):
    code = "E500"
