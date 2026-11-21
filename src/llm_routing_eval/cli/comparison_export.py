"""Atomic, no-PII export for the synthetic four-strategy comparison CLI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from llm_routing_eval.application import BenchmarkPersistenceError, ComparisonReport


def write_comparison(path: Path, report: ComparisonReport) -> None:
    """Never leave a partially written report; caller creates only safe DTOs."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".routing-comparison-",
            suffix=".json",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(report.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise BenchmarkPersistenceError("COMPARISON_EXPORT_FAILED") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
