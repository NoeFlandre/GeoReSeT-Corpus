"""CRAP score per function: complexity**2 * (1 - coverage)**3 + complexity.

Usage: ``python -m tools.crap coverage.json 6`` (fails when any function reaches 6).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from radon.complexity import cc_visit


@dataclass(frozen=True)
class Function:
    path: str
    name: str
    line: int
    complexity: int
    coverage: float


def crap(complexity: int, coverage: float) -> float:
    return complexity**2 * (1 - coverage) ** 3 + complexity


def functions(cov: dict) -> list[Function]:
    """Every function in the coverage report, with its complexity and line coverage."""
    out = []
    for path, data in cov["files"].items():
        executed, missing = set(data["executed_lines"]), set(data["missing_lines"])
        for block in cc_visit(Path(path).read_text()):
            lines = set(range(block.lineno, block.endline + 1))
            statements = lines & (executed | missing)
            covered = len(statements & executed) / len(statements) if statements else 1.0
            out.append(Function(path, block.fullname, block.lineno, block.complexity, covered))
    return out


def _line(score: float, f: Function) -> str:
    return f"CRAP {score:5.1f}  {f.path}:{f.line} {f.name}"


def report(fns: list[Function], threshold: float) -> tuple[list[str], float]:
    """Lines for functions at or over ``threshold``, worst first, and the worst score."""
    scores = sorted(((crap(f.complexity, f.coverage), f) for f in fns), key=lambda s: -s[0])
    lines = [_line(score, f) for score, f in scores if score >= threshold]
    return lines, scores[0][0] if scores else 0.0


def main(argv: list[str]) -> int:
    cov_path, threshold = argv[0], float(argv[1])
    lines, worst = report(functions(json.loads(Path(cov_path).read_text())), threshold)
    print("\n".join([*lines, f"max CRAP {worst:.1f} (threshold {threshold:g})"]))
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
