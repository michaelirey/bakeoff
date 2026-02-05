#!/usr/bin/env python3
"""Bakeoff orchestrator (KISS, stateful).

This script manages:
- per-target run state (JSON)
- git worktrees + branches
- GitHub PR discovery/cleanup

It does NOT directly invoke OpenClaw tools (scripts are plain Python), but it *does* produce
machine-readable action plans for OpenClaw to execute with PTY/background sessions.

In practice:
- `start` writes per-agent prompt files and emits an action plan (spawn 3 workers).
- `tick` can emit follow-on action plans (spawn missing workers, spawn review jobs).

Workflow:
- start: create run, worktrees, prompts, emit spawn-worker actions
- tick: discover PRs, advance to reviews, emit actions as needed
- merge: merge winner PR, close others, cleanup branches/worktrees, release lock

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

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
LOCK_HELPER = ROOT / "scripts" / "bakeoff_lock.py"

AGENTS = ("codex", "claude", "gemini")


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


def openclaw_exec_action(label: str, workdir: Path, command: str) -> Dict[str, Any]:
    """Machine-readable action: OpenClaw should run functions.exec with these params."""
    return {
        "kind": "openclaw.exec",
        "label": label,
        "workdir": str(workdir),
        "pty": True,
        "background": True,
        "timeoutSeconds": 3600,
        "command": command,
    }


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

    state = State(
        {
            "run_id": run_id,
            "phase": "phase1_prs",
            "target": {
                "repo_path": str(repo_path),
                "repo_url": args.repo_url or "",
                "base_branch": args.base_branch or "main",
                "base_ref": base_ref,
                "base_oid": git_current_main_oid(repo_path, base_ref),
            },
            "task": args.task,
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

    # Emit action plan for OpenClaw + also printable shell commands
    actions = []
    shell_commands = {}
    for a in AGENTS:
        wt = Path(agents[a]["worktree"])
        pf = Path(agents[a]["prompt_file"])
        cmd = agent_shell_command(a, pf, model_overrides)
        shell_commands[a] = f"cd {wt} && {cmd}"
        actions.append(openclaw_exec_action(f"phase1:{a}", wt, cmd))

    print(json.dumps({
        "run_id": run_id,
        "phase": state.phase,
        "shell": shell_commands,
        "actions": actions,
        "note": "Recommended: have OpenClaw execute `actions` with pty:true + background:true and store returned sessionIds into state (next step)."
    }, indent=2))

    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)
    st = load_state(repo_path)
    if not st:
        raise SystemExit("No state found; run start first")

    phase = st.phase
    if phase == "phase1_prs":
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
            st.data["phase"] = "phase2_reviews"
            st.data["timing"]["phase1_done_at"] = now_epoch()
            # Initialize reviews matrix and mark not-started
            st.data["reviews"] = {a: {"reviewed": {b: False for b in AGENTS if b != a}} for a in AGENTS}
            st.data["reviews_started_at"] = None
            changed = True
        if changed:
            st.save(state_path(repo_path))
        print(json.dumps(st.data, indent=2))
        return 0

    if phase == "phase2_reviews":
        # Emit phase2 actions once.
        actions = []
        shell = {}
        if not st.data.get("reviews_started_at"):
            st.data["reviews_started_at"] = now_epoch()
            # Write review prompts
            prompts_dir = run_dir(repo_path) / "prompts" / st.data["run_id"]
            for reviewer in AGENTS:
                targets = [st.data["agents"][a]["pr"] for a in AGENTS if a != reviewer]
                if len(targets) != 2:
                    raise SystemExit("Expected exactly 2 other PRs for cross-review")

                tmpl = load_template("CROSS_REVIEW.md")
                prompt_text = render_template(
                    tmpl,
                    {
                        "RUN_ID": st.data["run_id"],
                        "REVIEWER_AGENT": reviewer,
                        "MODEL_LABEL": st.data.get("models", {}).get(reviewer, ""),
                        "REPO_URL": st.data.get("target", {}).get("repo_url", ""),
                        "PR_A_URL": targets[0],
                        "PR_B_URL": targets[1],
                    },
                )
                pf = prompts_dir / f"review-{reviewer}.txt"
                write_prompt(pf, prompt_text)
                st.data.setdefault("review_prompts", {})[reviewer] = str(pf)

            st.save(state_path(repo_path))

        # Build actions from prompts
        model_overrides = {
            "codex": st.data.get("models", {}).get("codex", "gpt-5.2-codex"),
            "claude": st.data.get("models", {}).get("claude", "opus"),
            "gemini": st.data.get("models", {}).get("gemini", "gemini-3-pro-preview"),
        }
        for reviewer in AGENTS:
            wt = Path(st.data["agents"][reviewer]["worktree"])
            pf = Path(st.data["review_prompts"][reviewer])
            cmd = agent_shell_command(reviewer, pf, model_overrides)
            shell[reviewer] = f"cd {wt} && {cmd}"
            actions.append(openclaw_exec_action(f"phase2:{reviewer}", wt, cmd))

        print(json.dumps({"state": st.data, "shell": shell, "actions": actions}, indent=2))
        print("\nPhase 2 active. After review comments are posted, mark completion with: bakeoff.py mark-review ...")
        return 0

    if phase == "phase3_merge":
        print(json.dumps(st.data, indent=2))
        return 0

    print(json.dumps(st.data, indent=2))
    return 0


def cmd_mark_review(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    st = load_state(repo_path)
    if not st:
        raise SystemExit("No state")
    if st.phase != "phase2_reviews":
        raise SystemExit(f"Not in phase2_reviews (phase={st.phase})")
    reviewer = args.reviewer
    target = args.target
    st.data["reviews"][reviewer]["reviewed"][target] = True
    # If all True, advance.
    all_done = True
    for r in AGENTS:
        for t in AGENTS:
            if t == r:
                continue
            if not st.data["reviews"][r]["reviewed"][t]:
                all_done = False
    if all_done:
        st.data["phase"] = "phase3_merge"
        st.data["timing"]["phase2_done_at"] = now_epoch()
    st.save(state_path(repo_path))
    print(json.dumps(st.data, indent=2))
    return 0


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
    action = openclaw_exec_action(f"issue_select:{agent}", repo_path, cmd)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "agent": agent,
                "prompt_file": str(pf),
                "shell": f"cd {repo_path} && {cmd}",
                "actions": [action],
            },
            indent=2,
        )
    )
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
    action = openclaw_exec_action(f"merge_reco:{agent}", repo_path, cmd)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "agent": agent,
                "prompt_file": str(pf),
                "shell": f"cd {repo_path} && {cmd}",
                "actions": [action],
            },
            indent=2,
        )
    )
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

    mr = sub.add_parser("mark-review", help="Mark that reviewer commented on target PR")
    mr.add_argument("--repo-path", required=True)
    mr.add_argument("--reviewer", choices=AGENTS, required=True)
    mr.add_argument("--target", choices=AGENTS, required=True)
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
