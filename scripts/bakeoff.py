#!/usr/bin/env python3
"""Bakeoff orchestrator (KISS, stateful).

This script manages:
- per-target run state (JSON)
- git worktrees + branches
- GitHub PR discovery/cleanup

It runs locally as a plain Python script and is designed to be driven by a simple loop
(e.g. `while true; do bakeoff tick; sleep 60; done`). It persists state so it can resume
where it left off.

Modes:
- supervised: cross-review → human merge (original behavior)
- dark: behavioral validation → LLM-as-judge → auto-merge (dark factory)

Workflow (supervised):
- start → phase1_prs → phase2_reviews → phase3_merge (human)
Workflow (dark factory):
- start → phase1_prs → phase2_validate → phase3_converge → auto-merge

State is stored under: runs/<repo-slug>/state.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
LOCK_HELPER = ROOT / "scripts" / "bakeoff_lock.py"
CONFIG_PATH = ROOT / "config.yaml"

AGENTS = ("codex", "claude", "gemini")


def load_config() -> Dict[str, Any]:
    """Load config.yaml, returning empty dict on failure."""
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def is_dark_mode(config: Optional[Dict[str, Any]] = None, state: Optional[State] = None) -> bool:
    """Check if dark factory mode is enabled (per-run override takes precedence)."""
    if state and state.data.get("factory_mode") == "dark":
        return True
    if state and state.data.get("factory_mode") == "supervised":
        return False
    cfg = config or load_config()
    return cfg.get("factory", {}).get("mode", "supervised") == "dark"


def factory_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return factory config block with defaults."""
    cfg = config or load_config()
    fc = cfg.get("factory", {})
    return {
        "mode": fc.get("mode", "supervised"),
        "satisfaction_threshold": fc.get("satisfaction_threshold", 0.8),
        "max_convergence_rounds": fc.get("max_convergence_rounds", 3),
        "holdout_scenarios_dir": fc.get("holdout_scenarios_dir", "./scenarios"),
        "validation": {
            "run_tests": fc.get("validation", {}).get("run_tests", True),
            "test_command": fc.get("validation", {}).get("test_command", ""),
            "llm_judge": fc.get("validation", {}).get("llm_judge", True),
            "judge_model": fc.get("validation", {}).get("judge_model", "opus"),
        },
        "spec_enrichment": {
            "enabled": fc.get("spec_enrichment", {}).get("enabled", True),
            "enricher_model": fc.get("spec_enrichment", {}).get("enricher_model", "opus"),
        },
    }


def sh(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def repo_slug(repo_path: Path) -> str:
    p = repo_path.expanduser().resolve()
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
    return f"{p.name}-{h}"


def run_dir(repo_path: Path) -> Path:
    return RUNS_DIR / repo_slug(repo_path)


def state_path(repo_path: Path) -> Path:
    return run_dir(repo_path) / "state.json"


def now_epoch() -> int:
    return int(time.time())


@dataclass
class State:
    data: Dict[str, Any]

    @property
    def phase(self) -> str:
        return self.data.get("phase", "")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2) + "\n")


def load_state(repo_path: Path) -> Optional[State]:
    sp = state_path(repo_path)
    if not sp.exists():
        return None
    return State(json.loads(sp.read_text()))


def ensure_git_repo(repo_path: Path) -> None:
    if not (repo_path / ".git").exists():
        raise SystemExit(f"Not a git repo: {repo_path}")


def git_current_main_oid(repo_path: Path, base_ref: str) -> str:
    r = sh(["git", "rev-parse", base_ref], cwd=repo_path)
    return r.stdout.strip()


def write_prompt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


TEMPLATES_DIR = ROOT / "playbook" / "templates"


def render_template(text: str, vars: Dict[str, str]) -> str:
    """Very small {{VAR}} renderer."""
    out = text
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def load_template(name: str) -> str:
    path = TEMPLATES_DIR / name
    return path.read_text()


def agent_shell_command(agent: str, prompt_file: Path, model_overrides: Dict[str, str]) -> str:
    """Return a shell command intended to be run from within the agent worktree.

    We keep workdir separate so OpenClaw can run it with `workdir=...`.

    Notes:
    - Claude: prefer `-p` to reduce TUI noise; tool use still works.
    - Gemini: must use `-p` for headless mode.
    """
    if agent == "codex":
        model = model_overrides.get("codex", "gpt-5.2-codex")
        # Codex CLI supports reading the prompt from stdin by passing PROMPT as `-`.
        # Use `-C .` to anchor the workspace root to the current workdir.
        return f"cat {prompt_file} | codex exec -C . --dangerously-bypass-approvals-and-sandbox -m {model} -"
    if agent == "claude":
        model = model_overrides.get("claude", "opus")
        # Feed the prompt on stdin to avoid shell quoting/escaping issues.
        # `-p ""` keeps Claude in headless mode while still consuming stdin.
        return (
            f"cat {prompt_file} | CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1 "
            f"claude --model {model} --dangerously-skip-permissions "
            f"--permission-mode bypassPermissions -p \"\""
        )
    if agent == "gemini":
        model = model_overrides.get("gemini", "gemini-3-pro-preview")
        # Same stdin pattern; `-p ""` ensures headless execution.
        return f"cat {prompt_file} | gemini --yolo -m {model} -p \"\""
    raise ValueError(agent)


def spawn_pty_background(command: str, cwd: Path, log_path: Path) -> int:
    """Spawn a background process with a PTY (portable, no `script` dependency).

    Returns pid. Output is appended to log_path.
    """
    import pty

    log_path.parent.mkdir(parents=True, exist_ok=True)

    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(str(cwd))
        os.execvp("/bin/bash", ["/bin/bash", "-lc", command])
        raise SystemExit(1)

    # Parent: stream PTY output to log in a child logger process.
    f = open(log_path, "ab", buffering=0)

    logger_pid = os.fork()
    if logger_pid == 0:
        try:
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                f.write(chunk)
        finally:
            try:
                f.close()
            except Exception:
                pass
        os._exit(0)

    # Return the main command PID; logger is best-effort.
    return pid


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _model_branch_prefix(model: str) -> str:
    """Return a git-branch-safe model prefix for naming branches."""
    # Prefer stable, human-friendly prefixes.
    # - Claude: user uses `--model opus`, but we prefix as opus-4-5.
    if model.strip() == "opus":
        return "opus-4-5"
    # Git branch names don't love dots in some tooling; normalize to dashes.
    return re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()


def create_worktrees(
    repo_path: Path,
    run_id: str,
    base_ref: str,
    model_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Create worktrees and branches. Returns agent map with worktree+branch."""
    repo_path = repo_path.resolve()
    out: Dict[str, Dict[str, str]] = {}
    model_overrides = model_overrides or {}

    for agent in AGENTS:
        model = model_overrides.get(agent, "")
        prefix = _model_branch_prefix(model) if model else agent
        # Include agent suffix for uniqueness across parallel workers.
        branch = f"exp/{prefix}-bakeoff-{run_id}-{agent}"
        wt = repo_path.parent / f"wt-bakeoff-{repo_path.name}-{agent}-{run_id}"
        sh(["git", "worktree", "add", "-b", branch, str(wt), base_ref], cwd=repo_path)
        out[agent] = {"branch": branch, "worktree": str(wt), "model": model}
    return out


def gh_find_pr_by_head(repo_path: Path, head: str) -> Optional[str]:
    # `--head` uses current repo context.
    r = sh(["gh", "pr", "list", "--head", head, "--json", "url", "--limit", "1"], cwd=repo_path, check=False)
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except Exception:
        return None
    if isinstance(data, list) and data:
        return data[0].get("url")
    return None


def cmd_start(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    base_ref = f"origin/{args.base_branch}" if args.base_branch else "origin/main"

    # Acquire lock
    lock = sh([str(LOCK_HELPER), "acquire", "--repo", str(repo_path), "--note", f"bakeoff start {run_id}"], cwd=ROOT, check=False)
    if lock.returncode != 0:
        sys.stdout.write(lock.stdout)
        sys.stderr.write(lock.stderr)
        return lock.returncode

    # Ensure base ref exists
    sh(["git", "fetch", "origin"], cwd=repo_path)

    # NOTE: Models are used both for command generation and branch naming.
    model_overrides = {
        "codex": args.codex_model,
        "claude": args.claude_model,
        "gemini": args.gemini_model,
    }

    agents = create_worktrees(repo_path, run_id, base_ref, model_overrides=model_overrides)

    rd = run_dir(repo_path)
    prompts_dir = rd / "prompts" / run_id

    for agent in AGENTS:
        branch = agents[agent]["branch"]
        model_label = model_overrides.get(agent, "")

        if args.prompt_kind == "smoke":
            tmpl = load_template("SMOKE_TASK.md")
            prompt_text = render_template(
                tmpl,
                {
                    "RUN_ID": run_id,
                    "AGENT": agent,
                    "REPO_URL": args.repo_url or "",
                    "BASE_BRANCH": args.base_branch or "main",
                    "BRANCH_NAME": branch,
                    "TASK": args.task,
                },
            )
        else:
            tmpl = load_template("WORKER_TASK.md")
            prompt_text = render_template(
                tmpl,
                {
                    "RUN_ID": run_id,
                    "AGENT": agent,
                    "MODEL_LABEL": model_label,
                    "REPO_URL": args.repo_url or "",
                    "BASE_BRANCH": args.base_branch or "main",
                    "BRANCH_NAME": branch,
                    "ISSUE_NUMBER": str(args.issue_number or ""),
                    "ISSUE_URL": args.issue_url or "",
                    "PR_TITLE": args.pr_title or f"Bakeoff: issue #{args.issue_number}" if args.issue_number else "Bakeoff change",
                    "PR_BODY": args.pr_body or (f"Implements #{args.issue_number}." if args.issue_number else "Bakeoff change."),
                },
            )

        pf = prompts_dir / f"impl-{agent}.txt"
        write_prompt(pf, prompt_text)
        agents[agent]["prompt_file"] = str(pf)

    # Determine factory mode for this run
    run_mode = getattr(args, "mode", None) or factory_config().get("mode", "supervised")

    # Load scenarios (inline text or from file)
    scenarios_text = ""
    scenarios_arg = getattr(args, "scenarios", None)
    if scenarios_arg:
        scenarios_path = Path(scenarios_arg)
        if scenarios_path.exists():
            scenarios_text = scenarios_path.read_text()
        else:
            scenarios_text = scenarios_arg

    state = State(
        {
            "run_id": run_id,
            "phase": "phase1_prs",
            "factory_mode": run_mode,
            "target": {
                "repo_path": str(repo_path),
                "repo_url": args.repo_url or "",
                "base_branch": args.base_branch or "main",
                "base_ref": base_ref,
                "base_oid": git_current_main_oid(repo_path, base_ref),
            },
            "task": args.task,
            "scenarios": scenarios_text,
            "agents": {a: {"branch": agents[a]["branch"], "worktree": agents[a]["worktree"], "pr": None} for a in AGENTS},
            "models": {
                "codex": args.codex_model,
                "claude": args.claude_model,
                "gemini": args.gemini_model,
            },
            "timing": {"started_at": now_epoch(), "phase1_done_at": None, "phase2_done_at": None},
        }
    )
    state.save(state_path(repo_path))

    # Spawn workers immediately (this script is intended to be driven by a loop calling `tick`).
    logs_dir = run_dir(repo_path) / "logs" / run_id
    procs = {}

    for a in AGENTS:
        wt = Path(agents[a]["worktree"])
        pf = Path(agents[a]["prompt_file"])
        cmd = agent_shell_command(a, pf, model_overrides)
        log_path = logs_dir / f"phase1-{a}.log"
        pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)
        procs[a] = {"pid": pid, "log": str(log_path)}

    state.data["procs"] = {"phase1": procs}
    state.save(state_path(repo_path))

    print(json.dumps({
        "run_id": run_id,
        "phase": state.phase,
        "spawned": procs,
        "note": "Workers spawned. Run `bakeoff.py tick --repo-path ...` repeatedly (or via a loop) to advance phases."
    }, indent=2))

    return 0


def _parse_pr_map(st: State) -> Dict[str, Dict[str, Any]]:
    """Extract PR number and URL for each agent from state."""
    pr_map: Dict[str, Dict[str, Any]] = {}
    for a in AGENTS:
        url = st.data["agents"][a].get("pr")
        if not url:
            continue
        m = re.search(r"/pull/(\d+)$", url)
        if not m:
            raise SystemExit(f"Could not parse PR number from {url}")
        pr_map[a] = {"url": url, "num": int(m.group(1))}
    return pr_map


def _get_model_overrides(st: State) -> Dict[str, str]:
    """Extract model overrides from state."""
    return {
        "codex": st.data.get("models", {}).get("codex", "gpt-5.2-codex"),
        "claude": st.data.get("models", {}).get("claude", "opus"),
        "gemini": st.data.get("models", {}).get("gemini", "gemini-3-pro-preview"),
    }


def _tick_phase1_prs(st: State, repo_path: Path, config: Dict[str, Any]) -> None:
    """Phase 1: Discover PRs created by workers."""
    changed = False
    for agent in AGENTS:
        if st.data["agents"][agent].get("pr"):
            continue
        head = st.data["agents"][agent]["branch"]
        pr = gh_find_pr_by_head(repo_path, head)
        if pr:
            st.data["agents"][agent]["pr"] = pr
            changed = True

    if all(st.data["agents"][a].get("pr") for a in AGENTS):
        st.data["timing"]["phase1_done_at"] = now_epoch()
        changed = True

        if is_dark_mode(config, st):
            # Dark factory: skip reviews, go straight to validation
            st.data["phase"] = "phase2_validate"
            st.data["validation"] = {
                "started_at": None,
                "judge_jobs": {},
                "test_results": {},
                "reports": {},
                "convergence_round": 0,
            }
        else:
            # Supervised: go to cross-reviews
            st.data["phase"] = "phase2_reviews"
            st.data["reviews"] = {a: {"reviewed": {b: False for b in AGENTS if b != a}} for a in AGENTS}
            st.data["reviews_started_at"] = None

    if changed:
        st.save(state_path(repo_path))


def _tick_phase2_reviews(st: State, repo_path: Path) -> None:
    """Phase 2 (supervised mode): Cross-review jobs."""
    run_id = st.data["run_id"]
    st.data.setdefault("reviews_started_at", now_epoch())
    st.data.setdefault("review_jobs", {})

    pr_map = _parse_pr_map(st)
    model_overrides = _get_model_overrides(st)
    prompts_dir = run_dir(repo_path) / "prompts" / run_id
    logs_dir = run_dir(repo_path) / "logs" / run_id

    # Spawn any missing jobs
    for reviewer in AGENTS:
        for target in AGENTS:
            if reviewer == target:
                continue
            job_id = f"{reviewer}->{target}"
            job = st.data["review_jobs"].setdefault(job_id, {
                "reviewer": reviewer,
                "target": target,
                "pr_url": pr_map[target]["url"],
                "pr_num": pr_map[target]["num"],
                "pid": None,
                "log": None,
                "done": False,
            })

            if job.get("done"):
                continue
            pid = job.get("pid")
            if pid and pid_is_running(int(pid)):
                continue

            tmpl = load_template("CROSS_REVIEW_ONE.md")
            prompt_text = render_template(
                tmpl,
                {
                    "RUN_ID": run_id,
                    "REVIEWER_AGENT": reviewer,
                    "MODEL_LABEL": model_overrides.get(reviewer, ""),
                    "REPO_URL": st.data.get("target", {}).get("repo_url", ""),
                    "TARGET_PR_URL": pr_map[target]["url"],
                    "TARGET_PR_NUMBER": str(pr_map[target]["num"]),
                },
            )
            pf = prompts_dir / f"review-{reviewer}-on-{target}.md"
            write_prompt(pf, prompt_text)

            wt = Path(st.data["agents"][reviewer]["worktree"])
            cmd = agent_shell_command(reviewer, pf, model_overrides)
            log_path = logs_dir / f"phase2-{reviewer}-on-{target}.log"
            new_pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)
            job.update({"pid": new_pid, "log": str(log_path), "prompt": str(pf)})

    # Detect completion by checking PR comments for reviewer signature
    for job_id, job in st.data["review_jobs"].items():
        if job.get("done"):
            continue
        reviewer = job["reviewer"]
        pr_num = str(job["pr_num"])
        r = sh(["gh", "pr", "view", pr_num, "--json", "comments"], cwd=repo_path, check=False)
        if r.returncode != 0:
            continue
        try:
            data = json.loads(r.stdout)
        except Exception:
            continue
        comments = data.get("comments", [])
        if any(f"Reviewer: {reviewer}" in (c.get("body") or "") for c in comments):
            job["done"] = True
            job["done_at"] = now_epoch()

    if all(j.get("done") for j in st.data["review_jobs"].values()):
        st.data["phase"] = "phase3_merge"
        st.data["timing"]["phase2_done_at"] = now_epoch()

    st.save(state_path(repo_path))


def _tick_phase2_validate(st: State, repo_path: Path, config: Dict[str, Any]) -> None:
    """Phase 2 (dark factory mode): Behavioral validation via tests + LLM-as-judge."""
    from bakeoff_validate import (
        SatisfactionReport,
        failed_scenarios,
        format_failed_for_feedback,
        parse_judge_output,
        run_tests,
        select_winner,
    )

    run_id = st.data["run_id"]
    fc = factory_config(config)
    validation = st.data.setdefault("validation", {
        "started_at": now_epoch(),
        "judge_jobs": {},
        "test_results": {},
        "reports": {},
        "convergence_round": 0,
    })

    if not validation.get("started_at"):
        validation["started_at"] = now_epoch()

    pr_map = _parse_pr_map(st)
    model_overrides = _get_model_overrides(st)
    prompts_dir = run_dir(repo_path) / "prompts" / run_id
    logs_dir = run_dir(repo_path) / "logs" / run_id

    # Step 1: Run tests on each PR worktree (synchronous, quick)
    if fc["validation"]["run_tests"]:
        for agent in AGENTS:
            if agent in validation.get("test_results", {}):
                continue
            wt = Path(st.data["agents"][agent]["worktree"])
            test_cmd = fc["validation"]["test_command"] or None
            passed, output = run_tests(wt, test_cmd)
            validation.setdefault("test_results", {})[agent] = {
                "passed": passed,
                "output": output,
            }

    # Step 2: Spawn LLM-as-judge jobs for each PR
    if fc["validation"]["llm_judge"]:
        # Load scenarios from state (stored during spec enrichment or manually)
        scenarios_text = st.data.get("scenarios", "(no scenarios provided)")

        for agent in AGENTS:
            job_key = f"judge-{agent}"
            job = validation.setdefault("judge_jobs", {}).get(job_key)

            if job and job.get("done"):
                continue
            if job and job.get("pid") and pid_is_running(int(job["pid"])):
                continue

            pr_info = pr_map.get(agent, {})
            if not pr_info:
                continue

            tmpl = load_template("SATISFACTION_JUDGE.md")
            judge_model = fc["validation"]["judge_model"]
            prompt_text = render_template(
                tmpl,
                {
                    "RUN_ID": run_id,
                    "REPO_URL": st.data.get("target", {}).get("repo_url", ""),
                    "PR_URL": pr_info["url"],
                    "PR_NUMBER": str(pr_info["num"]),
                    "AGENT": agent,
                    "SCENARIOS": scenarios_text,
                },
            )
            pf = prompts_dir / f"judge-{agent}-r{validation.get('convergence_round', 0)}.md"
            write_prompt(pf, prompt_text)

            # Use the judge model agent (default: claude with judge_model)
            judge_overrides = dict(model_overrides)
            judge_overrides["claude"] = judge_model
            cmd = agent_shell_command("claude", pf, judge_overrides)
            log_path = logs_dir / f"judge-{agent}-r{validation.get('convergence_round', 0)}.log"
            pid = spawn_pty_background(cmd, cwd=repo_path, log_path=log_path)

            validation["judge_jobs"][job_key] = {
                "agent": agent,
                "pid": pid,
                "log": str(log_path),
                "done": False,
            }

        # Check for judge completion (look for JUDGE_DONE marker in logs)
        for job_key, job in validation.get("judge_jobs", {}).items():
            if job.get("done"):
                continue
            log_file = job.get("log")
            if not log_file or not Path(log_file).exists():
                continue
            try:
                log_content = Path(log_file).read_text(errors="replace")
            except Exception:
                continue
            if "JUDGE_DONE" in log_content:
                job["done"] = True
                job["done_at"] = now_epoch()
                # Parse the satisfaction report
                results = parse_judge_output(log_content)
                agent = job["agent"]
                test_passed = validation.get("test_results", {}).get(agent, {}).get("passed")
                report = SatisfactionReport(
                    agent=agent,
                    pr_number=pr_map.get(agent, {}).get("num", 0),
                    scenarios=results,
                    test_passed=test_passed,
                )
                validation.setdefault("reports", {})[agent] = report.to_dict()

    # Step 3: If all judges done, advance to convergence check
    judge_jobs = validation.get("judge_jobs", {})
    if judge_jobs and all(j.get("done") for j in judge_jobs.values()):
        st.data["phase"] = "phase3_converge"
        st.data["timing"]["phase2_done_at"] = now_epoch()

    st.save(state_path(repo_path))


def _tick_phase3_converge(st: State, repo_path: Path, config: Dict[str, Any]) -> None:
    """Phase 3 (dark factory mode): Check satisfaction, decide merge/iterate/reject."""
    fc = factory_config(config)
    threshold = fc["satisfaction_threshold"]
    max_rounds = fc["max_convergence_rounds"]
    validation = st.data.get("validation", {})
    current_round = validation.get("convergence_round", 0)

    reports = validation.get("reports", {})
    if not reports:
        # No reports yet — shouldn't happen, but be safe
        return

    # Find the best scoring agent
    best_agent = None
    best_score = -1.0
    for agent, report in reports.items():
        if report.get("test_passed") is False:
            continue  # Exclude agents with failed tests
        score = report.get("weighted_score", 0.0)
        if score > best_score:
            best_score = score
            best_agent = agent

    if best_agent and best_score >= threshold:
        # Winner found — auto-merge
        pr_map = _parse_pr_map(st)
        winner_pr = pr_map[best_agent]["num"]
        st.data["phase"] = "phase_auto_merge"
        st.data["winner"] = {
            "agent": best_agent,
            "pr": winner_pr,
            "score": best_score,
        }
        st.data["timing"]["converge_done_at"] = now_epoch()
        st.save(state_path(repo_path))
        return

    # Not satisfied — iterate or give up
    if current_round >= max_rounds:
        # Max rounds reached — escalate to human
        st.data["phase"] = "phase3_merge"  # Fall back to human merge
        st.data["escalation"] = {
            "reason": f"No agent reached satisfaction threshold {threshold} after {max_rounds} rounds",
            "best_agent": best_agent,
            "best_score": best_score,
        }
        st.save(state_path(repo_path))
        return

    # Generate convergence feedback and re-run
    run_id = st.data["run_id"]
    model_overrides = _get_model_overrides(st)
    prompts_dir = run_dir(repo_path) / "prompts" / run_id
    logs_dir = run_dir(repo_path) / "logs" / run_id
    pr_map = _parse_pr_map(st)

    next_round = current_round + 1
    validation["convergence_round"] = next_round

    # Spawn feedback + revision jobs for each agent
    for agent in AGENTS:
        report = reports.get(agent, {})
        failed = [s for s in report.get("scenarios", []) if s.get("verdict") in ("UNSATISFIED", "PARTIAL", "UNKNOWN")]
        if not failed:
            continue  # This agent is already satisfied

        failed_text = "\n".join(
            f"- {s['name']} (severity: {s['severity']}, verdict: {s['verdict']})\n  Reasoning: {s.get('reasoning', 'N/A')}"
            for s in failed
        )

        tmpl = load_template("CONVERGENCE_FEEDBACK.md")
        prompt_text = render_template(
            tmpl,
            {
                "RUN_ID": run_id,
                "AGENT": agent,
                "REPO_URL": st.data.get("target", {}).get("repo_url", ""),
                "BRANCH_NAME": st.data["agents"][agent]["branch"],
                "PR_URL": pr_map.get(agent, {}).get("url", ""),
                "PR_NUMBER": str(pr_map.get(agent, {}).get("num", "")),
                "ROUND": str(next_round),
                "MAX_ROUNDS": str(max_rounds),
                "FAILED_SCENARIOS": failed_text,
            },
        )
        pf = prompts_dir / f"converge-{agent}-r{next_round}.md"
        write_prompt(pf, prompt_text)

        wt = Path(st.data["agents"][agent]["worktree"])
        cmd = agent_shell_command(agent, pf, model_overrides)
        log_path = logs_dir / f"converge-{agent}-r{next_round}.log"
        pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)

        validation.setdefault("convergence_jobs", {})[f"converge-{agent}-r{next_round}"] = {
            "agent": agent,
            "round": next_round,
            "pid": pid,
            "log": str(log_path),
            "done": False,
        }

    # Reset judge jobs and reports for next round of validation
    validation["judge_jobs"] = {}
    validation["reports"] = {}
    st.data["phase"] = "phase2_validate"
    st.save(state_path(repo_path))


def _tick_phase_auto_merge(st: State, repo_path: Path) -> None:
    """Auto-merge the winner PR and clean up (dark factory mode)."""
    winner = st.data.get("winner", {})
    winner_pr = winner.get("pr")
    if not winner_pr:
        return

    # Merge winner
    sh(["gh", "pr", "merge", str(winner_pr), "--squash", "--delete-branch"], cwd=repo_path, check=False)

    # Close losing PRs
    for agent in AGENTS:
        pr_url = st.data["agents"][agent].get("pr")
        if not pr_url:
            continue
        m = re.search(r"/pull/(\d+)$", pr_url)
        if not m:
            continue
        pr_num = int(m.group(1))
        if pr_num == int(winner_pr):
            continue
        sh(["gh", "pr", "close", str(pr_num), "-c",
            f"Superseded by #{winner_pr} (auto-merged, satisfaction={winner.get('score', 'N/A')})."],
           cwd=repo_path, check=False)

    # Cleanup worktrees + branches
    for agent in AGENTS:
        wt = Path(st.data["agents"][agent]["worktree"]).expanduser()
        br = st.data["agents"][agent]["branch"]
        sh(["git", "worktree", "remove", str(wt), "--force"], cwd=repo_path, check=False)
        sh(["git", "branch", "-D", br], cwd=repo_path, check=False)
        sh(["git", "push", "origin", ":" + br], cwd=repo_path, check=False)

    # Release lock
    sh([str(LOCK_HELPER), "release", "--repo", str(repo_path)], cwd=ROOT, check=False)

    # Archive state
    st.data["phase"] = "done"
    st.data["merged_pr"] = int(winner_pr)
    st.data["timing"]["done_at"] = now_epoch()
    st.save(state_path(repo_path))


def cmd_tick(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)
    st = load_state(repo_path)
    if not st:
        raise SystemExit("No state found; run start first")

    config = load_config()
    phase = st.phase

    if phase == "phase1_prs":
        _tick_phase1_prs(st, repo_path, config)

    elif phase == "phase2_reviews":
        _tick_phase2_reviews(st, repo_path)

    elif phase == "phase2_validate":
        _tick_phase2_validate(st, repo_path, config)

    elif phase == "phase3_converge":
        _tick_phase3_converge(st, repo_path, config)

    elif phase == "phase_auto_merge":
        _tick_phase_auto_merge(st, repo_path)

    elif phase == "phase3_merge":
        pass  # Supervised mode: waiting for human `bakeoff merge`

    print(json.dumps(st.data, indent=2))
    return 0


def cmd_mark_review(args: argparse.Namespace) -> int:
    raise SystemExit("mark-review is deprecated; tick now auto-detects signed review comments.")


def cmd_merge(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)
    st = load_state(repo_path)
    if not st:
        raise SystemExit("No state")

    winner_pr = args.pr

    # Merge winner
    sh(["gh", "pr", "merge", str(winner_pr), "--squash", "--delete-branch"], cwd=repo_path)

    # Close others
    for agent in AGENTS:
        pr_url = st.data["agents"][agent].get("pr")
        if not pr_url:
            continue
        m = re.search(r"/pull/(\d+)$", pr_url)
        if not m:
            continue
        pr_num = int(m.group(1))
        if pr_num == int(winner_pr):
            continue
        sh(["gh", "pr", "close", str(pr_num), "-c", f"Superseded by #{winner_pr} (merged)."], cwd=repo_path, check=False)

    # Cleanup local worktrees + branches recorded in state
    for agent in AGENTS:
        wt = Path(st.data["agents"][agent]["worktree"]).expanduser()
        br = st.data["agents"][agent]["branch"]
        # Remove worktree
        sh(["git", "worktree", "remove", str(wt), "--force"], cwd=repo_path, check=False)
        # Delete local branch
        sh(["git", "branch", "-D", br], cwd=repo_path, check=False)
        # Delete remote branch best-effort
        sh(["git", "push", "origin", ":" + br], cwd=repo_path, check=False)

    # Release lock
    sh([str(LOCK_HELPER), "release", "--repo", str(repo_path)], cwd=ROOT, check=False)

    # Archive state
    st.data["phase"] = "done"
    st.data["merged_pr"] = int(winner_pr)
    st.data["timing"]["done_at"] = now_epoch()
    st.save(state_path(repo_path))

    print(json.dumps(st.data, indent=2))
    return 0


def cmd_select_issue(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    agent = args.agent

    rd = run_dir(repo_path)
    prompts_dir = rd / "prompts" / run_id
    tmpl = load_template("ISSUE_SELECTOR.md")
    prompt_text = render_template(
        tmpl,
        {
            "RUN_ID": run_id,
            "REPO_URL": args.repo_url or "",
            "BASE_BRANCH": args.base_branch or "main",
        },
    )
    pf = prompts_dir / f"issue-selector-{agent}.md"
    write_prompt(pf, prompt_text)

    model_overrides = {
        "codex": args.codex_model,
        "claude": args.claude_model,
        "gemini": args.gemini_model,
    }

    cmd = agent_shell_command(agent, pf, model_overrides)
    logs_dir = run_dir(repo_path) / "logs" / run_id
    log_path = logs_dir / f"issue-selector-{agent}.log"
    pid = spawn_pty_background(cmd, cwd=repo_path, log_path=log_path)

    print(json.dumps({
        "run_id": run_id,
        "agent": agent,
        "prompt_file": str(pf),
        "pid": pid,
        "log": str(log_path),
    }, indent=2))
    return 0


def cmd_recommend_merge(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    agent = args.agent

    rd = run_dir(repo_path)
    prompts_dir = rd / "prompts" / run_id

    tmpl = load_template("MERGE_STRATEGY.md")
    prompt_text = render_template(
        tmpl,
        {
            "RUN_ID": run_id,
            "REPO_URL": args.repo_url or "",
            "ISSUE_NUMBER": str(args.issue_number),
            "ISSUE_URL": args.issue_url or "",
            "PR_CODEX_URL": args.pr_codex_url,
            "PR_CLAUDE_URL": args.pr_claude_url,
            "PR_GEMINI_URL": args.pr_gemini_url,
        },
    )

    pf = prompts_dir / f"merge-recommendation-{agent}.md"
    write_prompt(pf, prompt_text)

    model_overrides = {
        "codex": args.codex_model,
        "claude": args.claude_model,
        "gemini": args.gemini_model,
    }

    cmd = agent_shell_command(agent, pf, model_overrides)
    logs_dir = run_dir(repo_path) / "logs" / run_id
    log_path = logs_dir / f"merge-recommendation-{agent}.log"
    pid = spawn_pty_background(cmd, cwd=repo_path, log_path=log_path)

    print(json.dumps({
        "run_id": run_id,
        "agent": agent,
        "prompt_file": str(pf),
        "pid": pid,
        "log": str(log_path),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bakeoff.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="Start a bakeoff run")
    s.add_argument("--repo-path", required=True)
    s.add_argument("--repo-url")
    s.add_argument("--task", required=True)
    s.add_argument("--prompt-kind", choices=("impl", "smoke"), default="impl")
    s.add_argument("--issue-number", type=int)
    s.add_argument("--issue-url")
    s.add_argument("--pr-title")
    s.add_argument("--pr-body")
    s.add_argument("--base-branch", default="main")
    s.add_argument("--mode", choices=("supervised", "dark"), help="Override factory mode for this run")
    s.add_argument("--scenarios", help="Acceptance scenarios text (or path to file) for dark factory validation")
    s.add_argument("--codex-model", default="gpt-5.2-codex")
    s.add_argument("--claude-model", default="opus")
    s.add_argument("--gemini-model", default="gemini-3-pro-preview")
    s.set_defaults(fn=cmd_start)

    si = sub.add_parser("select-issue", help="Emit an action plan to select the next issue")
    si.add_argument("--repo-path", required=True)
    si.add_argument("--repo-url")
    si.add_argument("--base-branch", default="main")
    si.add_argument("--agent", choices=AGENTS, default="codex")
    si.add_argument("--run-id")
    si.add_argument("--codex-model", default="gpt-5.2-codex")
    si.add_argument("--claude-model", default="opus")
    si.add_argument("--gemini-model", default="gemini-3-pro-preview")
    si.set_defaults(fn=cmd_select_issue)

    mr = sub.add_parser("recommend-merge", help="Emit an action plan for merge recommendation")
    mr.add_argument("--repo-path", required=True)
    mr.add_argument("--repo-url")
    mr.add_argument("--issue-number", type=int, required=True)
    mr.add_argument("--issue-url", required=True)
    mr.add_argument("--pr-codex-url", required=True)
    mr.add_argument("--pr-claude-url", required=True)
    mr.add_argument("--pr-gemini-url", required=True)
    mr.add_argument("--agent", choices=AGENTS, default="codex")
    mr.add_argument("--run-id")
    mr.add_argument("--codex-model", default="gpt-5.2-codex")
    mr.add_argument("--claude-model", default="opus")
    mr.add_argument("--gemini-model", default="gemini-3-pro-preview")
    mr.set_defaults(fn=cmd_recommend_merge)

    t = sub.add_parser("tick", help="Advance/check current run")
    t.add_argument("--repo-path", required=True)
    t.set_defaults(fn=cmd_tick)

    mr = sub.add_parser("mark-review", help="(deprecated) reviews are auto-detected in tick")
    mr.add_argument("--repo-path", required=True)
    mr.set_defaults(fn=cmd_mark_review)

    m = sub.add_parser("merge", help="Merge winner and cleanup")
    m.add_argument("--repo-path", required=True)
    m.add_argument("--pr", required=True, type=int)
    m.set_defaults(fn=cmd_merge)

    return ap


def main(argv: list[str]) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
