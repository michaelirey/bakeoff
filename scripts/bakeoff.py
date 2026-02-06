#!/usr/bin/env python3
"""Bakeoff orchestrator (KISS, stateful).

This script manages:
- per-target run state (JSON)
- git worktrees + branches
- GitHub PR discovery/cleanup

It runs locally as a plain Python script and is designed to be driven by a simple loop
(e.g. `while true; do bakeoff tick; sleep 60; done`). It persists state so it can resume
where it left off.

Workflow:
- start: create run, worktrees, prompts, emit spawn-worker actions
- tick: discover PRs, advance to reviews, emit actions as needed
- merge: merge winner PR, close others, cleanup branches/worktrees, release lock

State is stored under: runs/<repo-slug>/state.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shlex import quote as shlex_quote
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
LOCK_HELPER = ROOT / "scripts" / "bakeoff_lock.py"

AGENTS = ("codex", "claude", "gemini")


def sh(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def load_config(config_path: Optional[str | Path]) -> Dict[str, Any]:
    """Load bakeoff config.

    - If config_path is None, we try DEFAULT_CONFIG_PATH.
    - If file doesn't exist, return empty dict (preserve legacy behavior).

    NOTE: This is intended to be a refactor-only change: when config.yaml matches
    the legacy hardcoded defaults, behavior is unchanged.
    """

    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _cfg_get(cfg: Dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_cfg_path(p: str | Path) -> Path:
    p = str(p)
    return (ROOT / p).resolve() if not p.startswith("/") else Path(p).expanduser().resolve()


def runs_dir(cfg: Dict[str, Any]) -> Path:
    p = _cfg_get(cfg, ["paths", "runs_dir"], "./runs")
    return _resolve_cfg_path(p)


def logs_root(cfg: Dict[str, Any]) -> Path:
    p = _cfg_get(cfg, ["paths", "logs_dir"], "./logs")
    return _resolve_cfg_path(p)


def locks_root(cfg: Dict[str, Any]) -> Path:
    p = _cfg_get(cfg, ["paths", "locks_dir"], "./locks")
    return _resolve_cfg_path(p)


def templates_dir(cfg: Dict[str, Any]) -> Path:
    p = _cfg_get(cfg, ["templates", "dir"], "./playbook/templates")
    return _resolve_cfg_path(p)


def run_logs_dir(repo_path: Path, cfg: Dict[str, Any], run_id: str) -> Path:
    # Keep logs separate from runs so they can be rotated independently.
    # Structure: <logs_root>/<repo-slug>/<run_id>/...
    return logs_root(cfg) / repo_slug(repo_path) / run_id


def repo_slug(repo_path: Path) -> str:
    p = repo_path.expanduser().resolve()
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
    return f"{p.name}-{h}"


def run_dir(repo_path: Path, cfg: Dict[str, Any]) -> Path:
    return runs_dir(cfg) / repo_slug(repo_path)


def state_path(repo_path: Path, cfg: Dict[str, Any]) -> Path:
    return run_dir(repo_path, cfg) / "state.json"


# NOTE: legacy `state_path(repo_path)` removed; use `state_path(repo_path, cfg)`.


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


def load_state(repo_path: Path, cfg: Dict[str, Any]) -> Optional[State]:
    sp = state_path(repo_path, cfg)
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


# Template dir comes from config; this is just the legacy default.
# (Used only if config is missing / doesn't specify templates.dir)


def render_template(text: str, vars: Dict[str, str]) -> str:
    """Very small {{VAR}} renderer."""
    out = text
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def load_template(name: str, cfg: Dict[str, Any]) -> str:
    path = templates_dir(cfg) / name
    return path.read_text()


def _sanitize_role(role: str) -> str:
    # Filesystem-safe, deterministic.
    role = (role or "role").strip().lower()
    role = re.sub(r"[^a-z0-9]+", "-", role).strip("-")
    return role or "role"


def _mcp_rbac_role_for_bakeoff_role(bakeoff_role: str) -> str:
    """Map bakeoff workflow roles to bakeoff MCP server RBAC roles.

    The bakeoff MCP server enforces tool RBAC based on env BAKEOFF_MCP_ROLE:
    - admin: all tools
    - worker: bakeoff.publish only
    - author_revise: bakeoff.publish only
    - reviewer: bakeoff.comment only
    - verifier: bakeoff.comment only
    - orchestrator: bakeoff.comment + bakeoff.merge_and_cleanup

    Bakeoff role strings are more granular (e.g. "impl:codex", "review:claude-on-gemini").
    We map them to the smallest-privilege RBAC role that still allows the job.
    """
    r = (bakeoff_role or "").strip().lower()

    # Cross-reviews can only comment.
    if r.startswith("review:"):
        return "reviewer"

    # Manual verification can only comment.
    if r.startswith("manual_verify:"):
        return "verifier"

    # Merge recommendation is also comment-only.
    if r.startswith("merge_recommendation:"):
        return "verifier"

    # Implementation can publish only.
    if r.startswith("impl:"):
        return "worker"

    # Author revision can publish only.
    if r.startswith("author_revise:"):
        return "author_revise"

    # Issue selection should not need publish/merge; treat as orchestration.
    if r.startswith("issue_selection:"):
        return "orchestrator"

    # Default safest operational role for unknown jobs: orchestrator (no publish).
    return "orchestrator"


def _b64_path(p: Path) -> str:
    return base64.b64encode(str(p).encode("utf-8")).decode("ascii")


def _render_mcp_value(v: Any, vars: Dict[str, str]) -> Any:
    if isinstance(v, str):
        out = v
        for k, val in vars.items():
            out = out.replace("{" + k + "}", val)
        return out
    if isinstance(v, dict):
        return {k: _render_mcp_value(val, vars) for k, val in v.items()}
    if isinstance(v, list):
        return [_render_mcp_value(x, vars) for x in v]
    return v


def _mcp_servers_for_role(cfg: Dict[str, Any], *, run_id: str, role: str, agent: str, worktree: Path) -> Dict[str, Any]:
    raw = _cfg_get(cfg, ["mcp", "servers"], {})
    if not isinstance(raw, dict):
        return {}

    vars = {
        "run_id": run_id,
        "role": role,
        "agent": agent,
        "worktree": str(worktree),
        "worktree_b64": _b64_path(worktree),
    }

    out: Dict[str, Any] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        out[name] = _render_mcp_value(spec, vars)
    return out


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_codex_config_toml(path: Path, mcp_servers: Dict[str, Any]) -> None:
    """Write a minimal Codex CLI config.toml.

    Verified against a local Codex CLI install: MCP servers are configured under
    top-level `mcp_servers.<name>` tables (not `[mcp]` / `[mcp.servers]`).

    We intentionally only write keys we observe Codex using:
    - url (remote server)
    - command + args (stdio server)
    - env (nested table)
    - cwd (optional)
    """
    lines: list[str] = []
    lines.append('# Auto-generated by bakeoff.py (per run_id + role).')
    lines.append('')

    for name, spec in mcp_servers.items():
        if not isinstance(spec, dict):
            continue

        lines.append(f'[mcp_servers.{name}]')
        for k in ['url', 'command', 'args', 'cwd']:
            if k in spec:
                lines.append(f'{k} = {json.dumps(spec[k])}')

        env = spec.get('env')
        if isinstance(env, dict) and env:
            lines.append('')
            lines.append(f'[mcp_servers.{name}.env]')
            for ek, ev in env.items():
                lines.append(f'{json.dumps(str(ek))} = {json.dumps(str(ev))}')

        lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def agent_shell_command(
    agent: str,
    prompt_file: Path,
    model_overrides: Dict[str, str],
    repo_path: Path,
    run_id: str,
    role: str,
    worktree: Path,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a shell command intended to be run from within the agent worktree.

    Generates per-role MCP config files under runs/<repo>/mcp/<run_id>/<role>/.

    Role threading:
    - impl: impl:<agent>
    - review: review:<reviewer>-on-<target>
    - author_revise: author_revise:<agent>
    - manual_verify: manual_verify:<agent>
    - merge recommendation: merge_recommendation:<agent>

    Notes:
    - Claude: prefer `-p` to reduce TUI noise; tool use still works.
    - Gemini: must use `-p` for headless mode.
    """
    cfg = cfg or {}
    role_s = _sanitize_role(role)

    # Build MCP server config (optional).
    mcp_servers = _mcp_servers_for_role(cfg, run_id=run_id, role=role, agent=agent, worktree=worktree)
    allowed_names = sorted(mcp_servers.keys())

    mcp_dir = run_dir(repo_path, cfg) / 'mcp' / run_id / role_s
    extra_env: dict[str, str] = {}
    extra_args: list[str] = []

    # MCP servers are spawned by the agent CLI and inherit this environment.
    # Setting this ensures the bakeoff MCP server only *lists* and *allows* tools
    # permitted for the current job.
    extra_env['BAKEOFF_MCP_ROLE'] = _mcp_rbac_role_for_bakeoff_role(role)

    if mcp_servers:
        if agent == 'claude':
            mcp_path = mcp_dir / 'claude.mcp.json'
            _write_json(mcp_path, {'mcpServers': mcp_servers})
            extra_args += ['--strict-mcp-config', '--mcp-config', str(mcp_path)]
        elif agent == 'gemini':
            settings_path = mcp_dir / 'gemini.system-settings.json'
            _write_json(settings_path, {'mcpServers': mcp_servers})
            extra_env['GEMINI_CLI_SYSTEM_SETTINGS_PATH'] = str(settings_path)
            extra_args += ['--allowed-mcp-server-names', ','.join(allowed_names)]
        elif agent == 'codex':
            codex_home = mcp_dir / 'codex_home'
            config_path = codex_home / 'config.toml'
            _write_codex_config_toml(config_path, mcp_servers)
            extra_env['CODEX_HOME'] = str(codex_home)

    extra_env_prefix = ''.join(f"{k}={shlex_quote(str(v))} " for k, v in extra_env.items())

    if agent == "codex":
        cli = _cfg_get(cfg, ["agents", "codex", "cli"], "codex")
        model = model_overrides.get("codex") or _cfg_get(cfg, ["agents", "codex", "model"], "gpt-5.2-codex")
        # Codex CLI supports reading the prompt from stdin by passing PROMPT as `-`.
        # Use `-C .` to anchor the workspace root to the current workdir.
        return f"cat {prompt_file} | {extra_env_prefix}{cli} exec -C . --dangerously-bypass-approvals-and-sandbox -m {model} -"

    if agent == "claude":
        cli = _cfg_get(cfg, ["agents", "claude", "cli"], "claude")
        model = model_overrides.get("claude") or _cfg_get(cfg, ["agents", "claude", "model"], "opus")
        env = _cfg_get(cfg, ["agents", "claude", "env"], {})
        env_prefix = ''.join(f"{k}={shlex_quote(str(v))} " for k, v in env.items()) if isinstance(env, dict) else ""

        # Feed the prompt on stdin to avoid shell quoting/escaping issues.
        # `-p ""` keeps Claude in headless mode while still consuming stdin.
        extra = ' '.join(shlex_quote(a) for a in extra_args)
        extra = (extra + ' ') if extra else ''
        return (
            f"cat {prompt_file} | {extra_env_prefix}{env_prefix}"
            f"{cli} --model {model} --dangerously-skip-permissions "
            f"--permission-mode bypassPermissions {extra}-p \"\""
        )

    if agent == "gemini":
        cli = _cfg_get(cfg, ["agents", "gemini", "cli"], "gemini")
        model = model_overrides.get("gemini") or _cfg_get(cfg, ["agents", "gemini", "model"], "gemini-3-pro-preview")
        extra = ' '.join(shlex_quote(a) for a in extra_args)
        extra = (extra + ' ') if extra else ''
        # Same stdin pattern; `-p ""` ensures headless execution.
        return f"cat {prompt_file} | {extra_env_prefix}{cli} --yolo -m {model} {extra}-p \"\""

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
    """Return True if pid exists and is not a zombie.

    `os.kill(pid, 0)` returns success for zombies, so also consult `ps` state.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    r = sh(["ps", "-o", "state=", "-p", str(pid)], check=False)
    if r.returncode != 0:
        return False
    state = (r.stdout or "").strip()
    if not state:
        return False
    # Z = zombie
    return "Z" not in state


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


def gh_pr_has_comment(repo_path: Path, pr_num: str, needle: str) -> bool:
    """Best-effort check for an existing comment containing `needle`.

    Used to make tick idempotent even if a worker died after posting.
    """
    r = sh(["gh", "pr", "view", str(pr_num), "--json", "comments"], cwd=repo_path, check=False)
    if r.returncode != 0:
        return False
    try:
        data = json.loads(r.stdout)
    except Exception:
        return False
    comments = data.get("comments", []) if isinstance(data, dict) else []
    return any(needle in (c.get("body") or "") for c in comments)


def gh_pr_checks_ok(repo_path: Path, pr_num: int) -> Optional[bool]:
    """Return True if all checks succeeded, False if any failed, None if unknown.

    Uses gh's statusCheckRollup.
    """
    r = sh(["gh", "pr", "view", str(pr_num), "--json", "statusCheckRollup"], cwd=repo_path, check=False)
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except Exception:
        return None
    rollup = data.get("statusCheckRollup") if isinstance(data, dict) else None
    if not isinstance(rollup, list):
        return None

    any_failed = False
    any_pending = False
    for c in rollup:
        if not isinstance(c, dict):
            continue
        status = (c.get("status") or "").upper()
        concl = (c.get("conclusion") or "").upper()

        # If not completed, it's pending.
        if status and status != "COMPLETED":
            any_pending = True
            continue

        # Treat empty conclusion as pending/unknown.
        if not concl:
            any_pending = True
            continue

        if concl in {"FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"}:
            any_failed = True

    if any_failed:
        return False
    if any_pending:
        return None
    return True


def cmd_start(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    base_ref = f"origin/{args.base_branch}" if args.base_branch else "origin/main"

    # Acquire lock
    lock = sh([
        str(LOCK_HELPER),
        "acquire",
        "--repo",
        str(repo_path),
        "--locks-dir",
        str(locks_root(args.cfg)),
        "--note",
        f"bakeoff start {run_id}",
    ], cwd=ROOT, check=False)
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

    rd = run_dir(repo_path, args.cfg)
    prompts_dir = rd / "prompts" / run_id

    for agent in AGENTS:
        branch = agents[agent]["branch"]
        model_label = model_overrides.get(agent, "")

        if args.prompt_kind == "smoke":
            tmpl = load_template("SMOKE_TASK.md", args.cfg)
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
            tmpl = load_template("WORKER_TASK.md", args.cfg)
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
            "issue": {
                "number": args.issue_number,
                "url": args.issue_url or "",
            },
            "agents": {a: {"branch": agents[a]["branch"], "worktree": agents[a]["worktree"], "pr": None} for a in AGENTS},
            "models": {
                "codex": args.codex_model,
                "claude": args.claude_model,
                "gemini": args.gemini_model,
            },
            "timing": {"started_at": now_epoch(), "phase1_done_at": None, "phase2_done_at": None},
        }
    )
    state.save(state_path(repo_path, args.cfg))

    # Spawn workers immediately (this script is intended to be driven by a loop calling `tick`).
    logs_dir = run_logs_dir(repo_path, args.cfg, run_id)
    procs = {}

    for a in AGENTS:
        wt = Path(agents[a]["worktree"])
        pf = Path(agents[a]["prompt_file"])
        cmd = agent_shell_command(a, pf, model_overrides, repo_path, run_id, f"impl:{a}", wt, args.cfg)
        log_path = logs_dir / f"phase1-{a}.log"
        pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)
        procs[a] = {"pid": pid, "log": str(log_path)}

    state.data["procs"] = {"phase1": procs}
    state.save(state_path(repo_path, args.cfg))

    print(json.dumps({
        "run_id": run_id,
        "phase": state.phase,
        "spawned": procs,
        "note": "Workers spawned. Run `bakeoff.py tick --repo-path ...` repeatedly (or via a loop) to advance phases."
    }, indent=2))

    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)
    st = load_state(repo_path, args.cfg)
    if not st:
        raise SystemExit("No state found; run start first")

    phase = st.phase
    if phase == "phase1_prs":
        changed = False
        run_id = st.data["run_id"]

        # If a worker died before opening a PR, respawn it.
        st.data.setdefault("procs", {}).setdefault("phase1", {})
        logs_dir = run_logs_dir(repo_path, args.cfg, run_id)
        prompts_dir = run_dir(repo_path, args.cfg) / "prompts" / run_id

        model_overrides = {
            "codex": st.data.get("models", {}).get("codex", "gpt-5.2-codex"),
            "claude": st.data.get("models", {}).get("claude", "opus"),
            "gemini": st.data.get("models", {}).get("gemini", "gemini-3-pro-preview"),
        }

        for agent in AGENTS:
            if st.data["agents"][agent].get("pr"):
                continue

            # If we have a recorded pid but it's not running (or is a zombie), respawn.
            proc = st.data["procs"]["phase1"].get(agent) or {}
            pid = proc.get("pid")
            if pid and not pid_is_running(int(pid)):
                wt = Path(st.data["agents"][agent]["worktree"])

                # If the worktree is dirty, do NOT blindly respawn: the worker likely made progress
                # and exited early. Re-running the whole prompt from scratch can cause the agent
                # to halt on "unexpected changes".
                dirty = sh(["git", "status", "--porcelain"], cwd=wt, check=False).stdout.strip()
                if dirty:
                    st.data.setdefault("warnings", []).append({
                        "at": now_epoch(),
                        "agent": agent,
                        "kind": "phase1_worker_exited_dirty_worktree",
                        "message": "Worker process exited but worktree has uncommitted changes; not respawning automatically.",
                    })
                    # Record that the pid is dead for observability.
                    st.data["procs"]["phase1"][agent]["pid"] = None
                    changed = True
                else:
                    pf = prompts_dir / f"impl-{agent}.txt"
                    cmd = agent_shell_command(agent, pf, model_overrides, repo_path, run_id, f"impl:{agent}", wt, args.cfg)
                    log_path = logs_dir / f"phase1-{agent}.log"
                    new_pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)
                    st.data["procs"]["phase1"][agent] = {"pid": new_pid, "log": str(log_path), "respawned_at": now_epoch()}
                    changed = True

            head = st.data["agents"][agent]["branch"]
            pr = gh_find_pr_by_head(repo_path, head)
            if pr:
                st.data["agents"][agent]["pr"] = pr
                changed = True

        if all(st.data["agents"][a].get("pr") for a in AGENTS):
            reviews_enabled = bool(_cfg_get(args.cfg, ["workflow", "reviews", "enabled"], True))
            author_revise_enabled = bool(_cfg_get(args.cfg, ["workflow", "author_revision", "enabled"], True))

            if reviews_enabled:
                st.data["phase"] = "phase2_reviews"
                # Initialize reviews matrix and mark not-started
                st.data["reviews"] = {a: {"reviewed": {b: False for b in AGENTS if b != a}} for a in AGENTS}
                st.data["reviews_started_at"] = None
            else:
                # If reviews are disabled, skip directly to manual verification (or author revise if desired).
                st.data["phase"] = "phase2c_manual_verify" if not author_revise_enabled else "phase2b_author_revise"

            st.data["timing"]["phase1_done_at"] = now_epoch()
            changed = True

        if changed:
            st.save(state_path(repo_path, args.cfg))
        print(json.dumps(st.data, indent=2))
        return 0

    if phase == "phase2_reviews":
        """Spawn review jobs (one PR per run).

        We create a directed reviewer->target job for each pair (reviewer != target).
        """
        run_id = st.data["run_id"]
        st.data.setdefault("reviews_started_at", now_epoch())

        # Ensure job tables exist
        st.data.setdefault("review_jobs", {})

        # Parse PR numbers
        pr_map: Dict[str, Dict[str, Any]] = {}
        for a in AGENTS:
            url = st.data["agents"][a].get("pr")
            if not url:
                continue
            m = re.search(r"/pull/(\d+)$", url)
            if not m:
                raise SystemExit(f"Could not parse PR number from {url}")
            pr_map[a] = {"url": url, "num": int(m.group(1))}

        model_overrides = {
            "codex": st.data.get("models", {}).get("codex", "gpt-5.2-codex"),
            "claude": st.data.get("models", {}).get("claude", "opus"),
            "gemini": st.data.get("models", {}).get("gemini", "gemini-3-pro-preview"),
        }

        prompts_dir = run_dir(repo_path, args.cfg) / "prompts" / run_id
        logs_dir = run_logs_dir(repo_path, args.cfg, run_id)

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

                # Idempotency: if the expected review comment already exists, mark done and don't respawn.
                pr_num_s = str(job["pr_num"])
                needle = f"Reviewer: {reviewer}"
                if gh_pr_has_comment(repo_path, pr_num_s, needle):
                    job["done"] = True
                    job["done_at"] = now_epoch()
                    job["pid"] = None
                    continue

                pid = job.get("pid")
                if pid and pid_is_running(int(pid)):
                    continue

                # If previous pid exists but isn't running, we can retry.
                tmpl = load_template("CROSS_REVIEW_ONE.md", args.cfg)
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
                cmd = agent_shell_command(reviewer, pf, model_overrides, repo_path, run_id, f"review:{reviewer}-on-{target}", wt, args.cfg)
                log_path = logs_dir / f"phase2-{reviewer}-on-{target}.log"
                new_pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)
                job.update({"pid": new_pid, "log": str(log_path), "prompt": str(pf)})

        # Detect completion by checking PR comments for our signature.
        # Minimal heuristic: if the PR has a comment containing "Reviewer: <reviewer>", mark job done.
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

        # If all review jobs done, advance to author revision round (if enabled).
        if all(j.get("done") for j in st.data["review_jobs"].values()):
            author_revise_enabled = bool(_cfg_get(args.cfg, ["workflow", "author_revision", "enabled"], True))
            if author_revise_enabled:
                st.data["phase"] = "phase2b_author_revise"
            else:
                st.data["phase"] = "phase2c_manual_verify"
            st.data["timing"]["phase2_done_at"] = now_epoch()

        st.save(state_path(repo_path, args.cfg))
        print(json.dumps(st.data, indent=2))
        return 0

    if phase == "phase2b_author_revise":
        """Author revision round (one pass per PR author).

        Spawns one job per agent to incorporate (or explicitly acknowledge) review feedback on their own PR.
        Completion is detected by a PR comment containing "Author response (<agent>)".
        """
        run_id = st.data["run_id"]
        st.data.setdefault("author_revise_started_at", now_epoch())
        st.data.setdefault("author_revise_jobs", {})

        # Parse PR numbers
        pr_map: Dict[str, Dict[str, Any]] = {}
        for a in AGENTS:
            url = st.data["agents"][a].get("pr")
            if not url:
                continue
            m = re.search(r"/pull/(\d+)$", url)
            if not m:
                raise SystemExit(f"Could not parse PR number from {url}")
            pr_map[a] = {"url": url, "num": int(m.group(1))}

        model_overrides = {
            "codex": st.data.get("models", {}).get("codex", "gpt-5.2-codex"),
            "claude": st.data.get("models", {}).get("claude", "opus"),
            "gemini": st.data.get("models", {}).get("gemini", "gemini-3-pro-preview"),
        }

        prompts_dir = run_dir(repo_path, args.cfg) / "prompts" / run_id
        logs_dir = run_logs_dir(repo_path, args.cfg, run_id)

        issue_number = str(st.data.get("issue", {}).get("number") or st.data.get("issue_number") or st.data.get("issue") or "")
        issue_url = str(st.data.get("issue", {}).get("url") or st.data.get("issue_url") or "")

        # Spawn any missing author-revise jobs
        for agent in AGENTS:
            job = st.data["author_revise_jobs"].setdefault(agent, {
                "agent": agent,
                "pr_url": pr_map[agent]["url"],
                "pr_num": pr_map[agent]["num"],
                "pid": None,
                "log": None,
                "done": False,
            })

            if job.get("done"):
                continue

            pr_num = str(job["pr_num"])
            needle = f"Author response ({agent})"

            # Idempotency: if the comment already exists, mark done and do not respawn.
            if gh_pr_has_comment(repo_path, pr_num, needle):
                job["done"] = True
                job["done_at"] = now_epoch()
                job["pid"] = None
                continue

            pid = job.get("pid")
            if pid and pid_is_running(int(pid)):
                continue

            tmpl = load_template("AUTHOR_REVISE.md", args.cfg)
            prompt_text = render_template(
                tmpl,
                {
                    "RUN_ID": run_id,
                    "AGENT": agent,
                    "PR_URL": pr_map[agent]["url"],
                    "PR_NUMBER": str(pr_map[agent]["num"]),
                    "ISSUE_NUMBER": issue_number,
                    "ISSUE_URL": issue_url,
                },
            )
            pf = prompts_dir / f"author-revise-{agent}.md"
            write_prompt(pf, prompt_text)

            wt = Path(st.data["agents"][agent]["worktree"])
            cmd = agent_shell_command(agent, pf, model_overrides, repo_path, run_id, f"author_revise:{agent}", wt, args.cfg)
            log_path = logs_dir / f"phase2b-author-revise-{agent}.log"
            new_pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)
            job.update({"pid": new_pid, "log": str(log_path), "prompt": str(pf)})

        # Detect completion by checking PR comments for our author response header.
        for agent, job in st.data["author_revise_jobs"].items():
            if job.get("done"):
                continue
            pr_num = str(job["pr_num"])
            needle = f"Author response ({agent})"
            if gh_pr_has_comment(repo_path, pr_num, needle):
                job["done"] = True
                job["done_at"] = now_epoch()

        if all(j.get("done") for j in st.data["author_revise_jobs"].values()):
            st.data["phase"] = "phase2c_manual_verify"

        st.save(state_path(repo_path, args.cfg))
        print(json.dumps(st.data, indent=2))
        return 0

    if phase == "phase2c_manual_verify":
        """Manual verification pass (human-like CLI check) before merge selection.

        For now we run a single verifier (codex) against each candidate PR, executed inside
        the target PR's worktree to avoid branch switching.

        Completion is detected by a PR comment containing "Manual verification (codex)".
        """
        run_id = st.data["run_id"]
        st.data.setdefault("manual_verify_started_at", now_epoch())
        st.data.setdefault("manual_verify_jobs", {})

        # Parse PR numbers
        pr_map: Dict[str, Dict[str, Any]] = {}
        for a in AGENTS:
            url = st.data["agents"][a].get("pr")
            if not url:
                continue
            m = re.search(r"/pull/(\d+)$", url)
            if not m:
                raise SystemExit(f"Could not parse PR number from {url}")
            pr_map[a] = {"url": url, "num": int(m.group(1))}

        model_overrides = {
            "codex": st.data.get("models", {}).get("codex", "gpt-5.2-codex"),
            "claude": st.data.get("models", {}).get("claude", "opus"),
            "gemini": st.data.get("models", {}).get("gemini", "gemini-3-pro-preview"),
        }

        prompts_dir = run_dir(repo_path, args.cfg) / "prompts" / run_id
        logs_dir = run_logs_dir(repo_path, args.cfg, run_id)

        issue_number = str(st.data.get("issue", {}).get("number") or "")
        issue_url = str(st.data.get("issue", {}).get("url") or "")

        verifier = "codex"

        for target in AGENTS:
            job_id = f"{verifier}->{target}"
            job = st.data["manual_verify_jobs"].setdefault(job_id, {
                "verifier": verifier,
                "target": target,
                "pr_url": pr_map[target]["url"],
                "pr_num": pr_map[target]["num"],
                "pid": None,
                "log": None,
                "done": False,
            })

            if job.get("done"):
                continue

            pr_num = str(job["pr_num"])
            needle = f"Manual verification ({verifier})"

            # Idempotency: if verification comment already exists, mark done.
            if gh_pr_has_comment(repo_path, pr_num, needle):
                job["done"] = True
                job["done_at"] = now_epoch()
                job["pid"] = None
                continue

            pid = job.get("pid")
            if pid and pid_is_running(int(pid)):
                continue

            tmpl = load_template("MANUAL_VERIFY_ONE.md", args.cfg)
            prompt_text = render_template(
                tmpl,
                {
                    "RUN_ID": run_id,
                    "VERIFIER_AGENT": verifier,
                    "TARGET_PR_URL": pr_map[target]["url"],
                    "TARGET_PR_NUMBER": str(pr_map[target]["num"]),
                    "ISSUE_NUMBER": issue_number,
                    "ISSUE_URL": issue_url,
                },
            )
            pf = prompts_dir / f"manual-verify-{verifier}-on-{target}.md"
            write_prompt(pf, prompt_text)

            # Execute verification in the TARGET worktree so the branch is already checked out.
            wt = Path(st.data["agents"][target]["worktree"])
            cmd = agent_shell_command(verifier, pf, model_overrides, repo_path, run_id, f"manual_verify:{verifier}", wt, args.cfg)
            log_path = logs_dir / f"phase2c-manual-verify-{verifier}-on-{target}.log"
            new_pid = spawn_pty_background(cmd, cwd=wt, log_path=log_path)
            job.update({"pid": new_pid, "log": str(log_path), "prompt": str(pf)})

        # Detect completion by checking PR comments for the manual verification header.
        for job_id, job in st.data["manual_verify_jobs"].items():
            if job.get("done"):
                continue
            pr_num = str(job["pr_num"])
            verifier = job["verifier"]
            r = sh(["gh", "pr", "view", pr_num, "--json", "comments"], cwd=repo_path, check=False)
            if r.returncode != 0:
                continue
            try:
                data = json.loads(r.stdout)
            except Exception:
                continue
            comments = data.get("comments", [])
            needle = f"Manual verification ({verifier})"
            if any(needle in (c.get("body") or "") for c in comments):
                job["done"] = True
                job["done_at"] = now_epoch()

        if all(j.get("done") for j in st.data["manual_verify_jobs"].values()):
            st.data["phase"] = "phase3_merge"

        st.save(state_path(repo_path, args.cfg))
        print(json.dumps(st.data, indent=2))
        return 0

    if phase == "phase3_merge":
        # Automated mode: recommend + merge.
        human_in_loop = bool(_cfg_get(args.cfg, ["workflow", "merge", "human_in_loop"], False))
        if human_in_loop:
            print(json.dumps(st.data, indent=2))
            return 0

        run_id = st.data["run_id"]
        st.data.setdefault("merge_started_at", now_epoch())

        # 1) Spawn merge recommendation (once)
        st.data.setdefault("merge_reco", {"pid": None, "log": None, "prompt": None, "done": False, "winner_pr": None})
        reco = st.data["merge_reco"]

        if not reco.get("done"):
            pid = reco.get("pid")
            if not (pid and pid_is_running(int(pid))):
                # Build prompt
                prompts_dir = run_dir(repo_path, args.cfg) / "prompts" / run_id
                tmpl = load_template("MERGE_STRATEGY.md", args.cfg)
                prompt_text = render_template(
                    tmpl,
                    {
                        "RUN_ID": run_id,
                        "REPO_URL": st.data.get("target", {}).get("repo_url", ""),
                        "ISSUE_NUMBER": str(st.data.get("issue", {}).get("number") or ""),
                        "ISSUE_URL": str(st.data.get("issue", {}).get("url") or ""),
                        "PR_CODEX_URL": st.data["agents"]["codex"]["pr"],
                        "PR_CLAUDE_URL": st.data["agents"]["claude"]["pr"],
                        "PR_GEMINI_URL": st.data["agents"]["gemini"]["pr"],
                    },
                )
                pf = prompts_dir / "merge-recommendation-codex.md"
                write_prompt(pf, prompt_text)

                model_overrides = st.data.get("models", {})
                wt = Path(st.data["agents"]["codex"]["worktree"])
                cmd = agent_shell_command("codex", pf, model_overrides, repo_path, run_id, "merge_recommendation:codex", wt, args.cfg)
                logs_dir = run_logs_dir(repo_path, args.cfg, run_id)
                log_path = logs_dir / "merge-recommendation-codex.log"
                new_pid = spawn_pty_background(cmd, cwd=repo_path, log_path=log_path)
                reco.update({"pid": new_pid, "log": str(log_path), "prompt": str(pf)})

            # Detect completion by parsing log for MERGE_RECOMMENDATION block.
            log_path = reco.get("log")
            if log_path and Path(log_path).exists():
                txt = Path(log_path).read_text(errors="ignore")
                # The prompt itself contains a MERGE_RECOMMENDATION template; only treat it as done
                # when we see a concrete numeric winner_pr in the log (and use the last one).
                ms = re.findall(r"winner_pr:\s*(\d+)", txt)
                if ms:
                    reco["winner_pr"] = int(ms[-1])
                    reco["done"] = True
                    reco["done_at"] = now_epoch()

        # 2) If we have a winner, merge and finish.
        if reco.get("done") and reco.get("winner_pr") and not st.data.get("merged_pr"):
            # Require green checks before merging. If winner has failing checks,
            # fall back to another candidate with passing checks.
            candidates = [
                int(reco["winner_pr"]),
                int(re.search(r"/pull/(\d+)$", st.data["agents"]["claude"]["pr"]).group(1)),
                int(re.search(r"/pull/(\d+)$", st.data["agents"]["gemini"]["pr"]).group(1)),
                int(re.search(r"/pull/(\d+)$", st.data["agents"]["codex"]["pr"]).group(1)),
            ]
            seen = []
            for c in candidates:
                if c not in seen:
                    seen.append(c)

            chosen: Optional[int] = None
            pending_any = False
            for prn in seen:
                ok = gh_pr_checks_ok(repo_path, prn)
                if ok is True:
                    chosen = prn
                    break
                if ok is None:
                    pending_any = True

            if chosen is None:
                st.data.setdefault("warnings", []).append({
                    "at": now_epoch(),
                    "kind": "merge_blocked_checks",
                    "message": "No candidate PR has all checks passing yet; waiting.",
                    "winner_pr": int(reco["winner_pr"]),
                })
                st.save(state_path(repo_path, args.cfg))
                print(json.dumps(st.data, indent=2))
                return 0

            if chosen != int(reco["winner_pr"]):
                st.data.setdefault("warnings", []).append({
                    "at": now_epoch(),
                    "kind": "merge_winner_failed_checks_fallback",
                    "message": f"Merge recommendation winner had failing checks; falling back to PR #{chosen} with passing checks.",
                    "winner_pr": int(reco["winner_pr"]),
                    "chosen_pr": chosen,
                })

            do_merge(repo_path, st, args.cfg, chosen)
            st.save(state_path(repo_path, args.cfg))
            print(json.dumps(st.data, indent=2))
            return 0

        st.save(state_path(repo_path, args.cfg))
        print(json.dumps(st.data, indent=2))
        return 0

    print(json.dumps(st.data, indent=2))
    return 0


def cmd_mark_review(args: argparse.Namespace) -> int:
    raise SystemExit("mark-review is deprecated; tick now auto-detects signed review comments.")


def do_merge(repo_path: Path, st: State, cfg: Dict[str, Any], winner_pr: int) -> None:
    merge_method = _cfg_get(cfg, ["workflow", "merge", "method"], "squash")
    delete_remote_branches = bool(_cfg_get(cfg, ["workflow", "merge", "delete_remote_branches"], True))
    close_losers = bool(_cfg_get(cfg, ["workflow", "merge", "close_losers"], True))

    # Merge winner
    merge_args = ["gh", "pr", "merge", str(winner_pr)]
    if merge_method == "merge":
        merge_args.append("--merge")
    elif merge_method == "rebase":
        merge_args.append("--rebase")
    else:
        merge_args.append("--squash")

    if delete_remote_branches:
        merge_args.append("--delete-branch")

    # Merge can be a no-op if already merged (or can fail if branch deletion is blocked
    # by local worktrees). Treat merge as best-effort and continue cleanup.
    sh(merge_args, cwd=repo_path, check=False)

    # Close others
    if close_losers:
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
        if delete_remote_branches:
            sh(["git", "push", "origin", ":" + br], cwd=repo_path, check=False)

    # Release lock
    sh([
        str(LOCK_HELPER),
        "release",
        "--repo",
        str(repo_path),
        "--locks-dir",
        str(locks_root(cfg)),
    ], cwd=ROOT, check=False)

    # Archive state
    st.data["phase"] = "done"
    st.data["merged_pr"] = int(winner_pr)
    st.data.setdefault("timing", {})
    st.data["timing"]["done_at"] = now_epoch()


def cmd_merge(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)
    st = load_state(repo_path, args.cfg)
    if not st:
        raise SystemExit("No state")

    do_merge(repo_path, st, args.cfg, int(args.pr))
    st.save(state_path(repo_path, args.cfg))
    print(json.dumps(st.data, indent=2))
    return 0


def cmd_select_issue(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).expanduser().resolve()
    ensure_git_repo(repo_path)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    agent = args.agent

    rd = run_dir(repo_path, args.cfg)
    prompts_dir = rd / "prompts" / run_id
    tmpl = load_template("ISSUE_SELECTOR.md", args.cfg)
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

    cmd = agent_shell_command(agent, pf, model_overrides, repo_path, run_id, f"issue_selection:{agent}", repo_path, args.cfg)
    logs_dir = run_logs_dir(repo_path, args.cfg, run_id)
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

    rd = run_dir(repo_path, args.cfg)
    prompts_dir = rd / "prompts" / run_id

    tmpl = load_template("MERGE_STRATEGY.md", args.cfg)
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

    cmd = agent_shell_command(agent, pf, model_overrides, repo_path, run_id, f"merge_recommendation:{agent}", repo_path, args.cfg)
    logs_dir = run_logs_dir(repo_path, args.cfg, run_id)
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
    ap.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
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

    # Load config once and attach for subcommands.
    cfg = load_config(getattr(args, "config", None))
    setattr(args, "cfg", cfg)

    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
