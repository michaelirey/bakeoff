#!/usr/bin/env python3
"""Bakeoff validation harness for dark factory mode.

Provides behavioral validation of PRs via:
1. Test suite execution (hard gate)
2. LLM-as-judge scenario satisfaction scoring (soft scoring)

The satisfaction score (0.0-1.0) replaces code review as the quality signal.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioResult:
    name: str
    severity: str  # critical | high | medium | low
    verdict: str  # SATISFIED | PARTIAL | UNSATISFIED | UNKNOWN
    reasoning: str

    @property
    def weight(self) -> float:
        return {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5}.get(self.severity, 1.0)

    @property
    def score(self) -> float:
        return {"SATISFIED": 1.0, "PARTIAL": 0.5, "UNSATISFIED": 0.0, "UNKNOWN": 0.0}.get(self.verdict, 0.0)


@dataclass
class SatisfactionReport:
    agent: str
    pr_number: int
    scenarios: List[ScenarioResult] = field(default_factory=list)
    test_passed: Optional[bool] = None
    test_output: str = ""

    @property
    def weighted_score(self) -> float:
        if not self.scenarios:
            return 0.0
        total_weight = sum(s.weight for s in self.scenarios)
        if total_weight == 0:
            return 0.0
        return sum(s.score * s.weight for s in self.scenarios) / total_weight

    @property
    def recommendation(self) -> str:
        if self.test_passed is False:
            return "reject"
        score = self.weighted_score
        if score >= 0.8:
            return "merge"
        if score >= 0.5:
            return "iterate"
        return "reject"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "pr_number": self.pr_number,
            "test_passed": self.test_passed,
            "weighted_score": round(self.weighted_score, 3),
            "recommendation": self.recommendation,
            "scenarios": [
                {
                    "name": s.name,
                    "severity": s.severity,
                    "verdict": s.verdict,
                    "reasoning": s.reasoning,
                }
                for s in self.scenarios
            ],
        }


def detect_test_command(worktree: Path) -> Optional[str]:
    """Auto-detect the test command for a repo."""
    if (worktree / "Makefile").exists():
        makefile = (worktree / "Makefile").read_text()
        if re.search(r"^test:", makefile, re.MULTILINE):
            return "make test"
    if (worktree / "pyproject.toml").exists():
        return "uv run pytest --tb=short -q 2>&1 || pytest --tb=short -q 2>&1"
    if (worktree / "setup.py").exists() or (worktree / "setup.cfg").exists():
        return "pytest --tb=short -q 2>&1"
    if (worktree / "package.json").exists():
        return "npm test 2>&1"
    if (worktree / "Cargo.toml").exists():
        return "cargo test 2>&1"
    if (worktree / "go.mod").exists():
        return "go test ./... 2>&1"
    return None


def run_tests(worktree: Path, test_command: Optional[str] = None) -> tuple[bool, str]:
    """Run the repo test suite in a worktree. Returns (passed, output)."""
    cmd = test_command or detect_test_command(worktree)
    if not cmd:
        # No tests detected — pass by default (don't block on missing tests)
        return True, "(no test suite detected)"

    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            cwd=str(worktree),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,  # 5 minute timeout
        )
        passed = result.returncode == 0
        output = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "(test suite timed out after 300s)"
    except Exception as e:
        return False, f"(test execution error: {e})"


def parse_judge_output(output: str) -> List[ScenarioResult]:
    """Parse the SATISFACTION_REPORT block from LLM judge output."""
    results = []
    # Find scenario result blocks
    pattern = re.compile(
        r"- scenario:\s*(.+?)\n"
        r"\s*severity:\s*(\w+)\n"
        r"\s*verdict:\s*(\w+)\n"
        r"\s*reasoning:\s*(.+?)(?=\n\s*-\s*scenario:|\nSUMMARY:|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(output):
        results.append(
            ScenarioResult(
                name=m.group(1).strip(),
                severity=m.group(2).strip().lower(),
                verdict=m.group(3).strip(),
                reasoning=m.group(4).strip(),
            )
        )
    return results


def parse_weighted_score(output: str) -> Optional[float]:
    """Extract the weighted_score from judge output as a fallback."""
    m = re.search(r"weighted_score:\s*([\d.]+)", output)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def failed_scenarios(report: SatisfactionReport) -> List[ScenarioResult]:
    """Return scenarios that are not fully satisfied."""
    return [s for s in report.scenarios if s.verdict in ("UNSATISFIED", "PARTIAL", "UNKNOWN")]


def format_failed_for_feedback(scenarios: List[ScenarioResult]) -> str:
    """Format failed scenarios for the CONVERGENCE_FEEDBACK template."""
    lines = []
    for s in scenarios:
        lines.append(f"- {s.name} (severity: {s.severity}, verdict: {s.verdict})")
        lines.append(f"  Reasoning: {s.reasoning}")
    return "\n".join(lines) if lines else "(all scenarios satisfied)"


def select_winner(reports: Dict[str, SatisfactionReport]) -> Optional[str]:
    """Select the best agent based on satisfaction scores.

    Returns agent name or None if no agent meets the threshold.
    Agents with failed tests are excluded.
    """
    eligible = {
        agent: report
        for agent, report in reports.items()
        if report.test_passed is not False  # None (no tests) or True
    }
    if not eligible:
        return None
    return max(eligible, key=lambda a: eligible[a].weighted_score)
