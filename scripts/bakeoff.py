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


def base_impl_prompt(task: str, run_id: str, agent: str) -> str:
    # Avoid angle brackets; use PR_URL_HERE placeholder.
    return f"""You are in the agentic_search repo.

TASK: {task}

Hard requirements:
- Use pytest.
- Add at least 2 unit tests that exercise pure/local logic WITHOUT needing OPENAI_API_KEY.
- Ensure CLI help works without OPENAI_API_KEY (lazy OpenAI client init if needed).
- Add a GitHub Actions workflow that installs deps with uv and runs tests on push + PR.
- Keep changes minimal and idiomatic.
- Do NOT commit secrets. Confirm `git status --porcelain` before commit; ensure .env is untracked.

Deliverables (you must do all of these):
1) Implement + tests.
2) Ensure pytest is installed via project config and CI installs it.
3) Run: `uv sync` and `uv run python -m pytest -q`.
4) Commit + push.
5) Create PR via `gh pr create` against main.
6) After PR exists, run exactly:
   openclaw gateway call cron.wake --params '{{"text":"BAKEOFF_DONE run={run_id} agent={agent} pr=PR_URL_HERE","mode":"now"}}'
   (Replace PR_URL_HERE with the real PR URL.)

Do not stop early. Execute the git + gh steps yourself.
"""


def base_review_prompt(run_id: str, reviewer: str, targets: Dict[str, str]) -> str:
    """Prompt for cross-review comments.

    targets: map agent->PR URL
    """
    # We instruct the agent to post GitHub PR comments directly.
    # Avoid angle brackets in shell.
    lines = [
        "You are in the agentic_search repo.",
        "",
        "TASK: Cross-review the other PRs from this bakeoff run.",
        "",
        "Rules:",
        "- Be concrete: cite specific diffs/lines and possible failure modes.",
        "- Focus on correctness, tests, CI, maintainability.",
        "- Post your review as a PR COMMENT (not a formal PR review).",
        "",
        f"This run_id: {run_id}",
        f"You are reviewer: {reviewer}",
        "",
        "For each PR below:",
        "1) Read the diff: `gh pr diff <NUM>`",
        "2) Write a structured review comment (bullets: ✅ good / ⚠️ risks / 🧪 tests / 📌 suggestions)",
        "3) Post it: `gh pr comment <NUM> --body \"<your review text>\"`",
        "",
    ]

    for agent, url in targets.items():
        lines.append(f"- PR for agent={agent}: {url}")

    lines += [
        "",
        "After you have posted comments on ALL target PRs, run exactly:",
        f"openclaw gateway call cron.wake --params '{{\"text\":\"BAKEOFF_REVIEW_DONE run={run_id} reviewer={reviewer}\",\"mode\":\"now\"}}'",
        "",
        "Do not stop early.",
    ]

    return "\n".join(lines) + "\n"


def agent_shell_command(agent: str, prompt_file: Path, model_overrides: Dict[str, str]) -> str:
    """Return a shell command intended to be run from within the agent worktree.

    We keep workdir separate so OpenClaw can run it with `workdir=...`.

    Notes:
    - Claude: prefer `-p` to reduce TUI noise; tool use still works.
    - Gemini: must use `-p` for headless mode.
    """
    if agent == "codex":
        model = model_overrides.get("codex", "gpt-5.2-codex")
        return f"codex exec --dangerously-bypass-approvals-and-sandbox -m {model} \"$(cat {prompt_file})\""
    if agent == "claude":
        model = model_overrides.get("claude", "opus")
        # Feed the prompt on stdin to avoid shell quoting/escaping issues.
        # `-p ""` keeps Claude in headless mode while still consuming stdin.
        return (
            f"cat {prompt_file} | claude --model {model} --dangerously-skip-permissions "
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


def create_worktrees(repo_path: Path, run_id: str, base_ref: str) -> Dict[str, Dict[str, str]]:
    """Create worktrees and branches. Returns agent map with worktree+branch."""
    repo_path = repo_path.resolve()
    out: Dict[str, Dict[str, str]] = {}
    for agent in AGENTS:
        branch = f"exp/bakeoff-{run_id}-{agent}"
        wt = repo_path.parent / f"wt-bakeoff-{repo_path.name}-{agent}-{run_id}"
        sh(["git", "worktree", "add", "-b", branch, str(wt), base_ref], cwd=repo_path)
        out[agent] = {"branch": branch, "worktree": str(wt)}
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

    agents = create_worktrees(repo_path, run_id, base_ref)

    rd = run_dir(repo_path)
    prompts_dir = rd / "prompts" / run_id

    model_overrides = {
        "codex": args.codex_model,
        "claude": args.claude_model,
        "gemini": args.gemini_model,
    }

    for agent in AGENTS:
        prompt_text = base_impl_prompt(args.task, run_id, agent)
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
                targets = {a: st.data["agents"][a]["pr"] for a in AGENTS if a != reviewer}
                pf = prompts_dir / f"review-{reviewer}.txt"
                write_prompt(pf, base_review_prompt(st.data["run_id"], reviewer, targets))
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bakeoff.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="Start a bakeoff run")
    s.add_argument("--repo-path", required=True)
    s.add_argument("--repo-url")
    s.add_argument("--task", required=True)
    s.add_argument("--base-branch", default="main")
    s.add_argument("--codex-model", default="gpt-5.2-codex")
    s.add_argument("--claude-model", default="opus")
    s.add_argument("--gemini-model", default="gemini-3-pro-preview")
    s.set_defaults(fn=cmd_start)

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
