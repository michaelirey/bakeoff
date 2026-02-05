#!/usr/bin/env python3
"""Bakeoff MCP server (stdio transport).

Implements a small subset of the Model Context Protocol (MCP) over stdio:
- initialize
- tools/list
- tools/call

Tools are declared in `mcp_tools.yaml` (registry) and enforced by basic RBAC.

Notes
- This server is intentionally dependency-light; it does not require the `mcp` SDK.
- It shells out to `git` and `gh`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "mcp_tools.yaml"
BAKEOFF_SCRIPT = ROOT / "scripts" / "bakeoff.py"


# -----------------------
# JSON-RPC / MCP helpers
# -----------------------

def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _err(_id: Any, code: int, message: str, data: Any = None) -> None:
    payload: dict = {"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}
    if data is not None:
        payload["error"]["data"] = data
    _send(payload)


def _ok(_id: Any, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": _id, "result": result})


def _now() -> int:
    return int(time.time())


# -----------------------
# Shell helpers
# -----------------------

def sh(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def ensure_git_repo(repo_path: Path) -> None:
    if not (repo_path / ".git").exists():
        raise RuntimeError(f"Not a git repo: {repo_path}")


# -----------------------
# Bakeoff state helpers
# -----------------------

def repo_slug(repo_path: Path) -> str:
    import hashlib

    p = repo_path.expanduser().resolve()
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
    return f"{p.name}-{h}"


def state_path_for_repo(repo_path: Path) -> Path:
    # Must match bakeoff/scripts/bakeoff.py defaults.
    return ROOT / "runs" / repo_slug(repo_path) / "state.json"


def load_state(repo_path: Path) -> dict:
    sp = state_path_for_repo(repo_path)
    if not sp.exists():
        return {}
    return json.loads(sp.read_text())


def save_state(repo_path: Path, state: dict) -> None:
    sp = state_path_for_repo(repo_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(state, indent=2) + "\n")


# -----------------------
# RBAC
# -----------------------

def caller_role(params: dict) -> str:
    # Prefer explicit param (useful for testing), else env.
    return (params.get("_role") or os.environ.get("BAKEOFF_MCP_ROLE") or "admin").strip() or "admin"


def enforce_rbac(tool_name: str, registry: dict, role: str) -> None:
    allowed = (((registry.get("tools") or {}).get(tool_name) or {}).get("allowed_roles")) or []
    if allowed and role not in allowed:
        raise PermissionError(f"Role '{role}' is not allowed to call {tool_name} (allowed: {allowed})")


# -----------------------
# GitHub helpers
# -----------------------

def gh_find_pr_by_head(repo_path: Path, head: str) -> Optional[dict]:
    r = sh(["gh", "pr", "list", "--head", head, "--json", "number,url", "--limit", "1"], cwd=repo_path, check=False)
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except Exception:
        return None
    if isinstance(data, list) and data:
        return {"number": data[0].get("number"), "url": data[0].get("url")}
    return None


def gh_pr_checks_ok(repo_path: Path, pr_number: int) -> Optional[bool]:
    """Return True if all checks succeeded, False if any failed, None if pending/unknown."""
    r = sh(["gh", "pr", "view", str(pr_number), "--json", "statusCheckRollup"], cwd=repo_path, check=False)
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

        if status and status != "COMPLETED":
            any_pending = True
            continue
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


# -----------------------
# Tool implementations
# -----------------------

def tool_publish(args: dict) -> dict:
    repo_path = Path(args["repo_path"]).expanduser().resolve()
    ensure_git_repo(repo_path)

    base_branch = args.get("base_branch") or "main"
    remote = args.get("remote") or "origin"
    pr_title = args["pr_title"].strip()
    pr_body = (args.get("pr_body") or "").strip()
    commit_message = (args.get("commit_message") or pr_title).strip()
    draft = bool(args.get("draft", False))
    allow_empty = bool(args.get("allow_empty", False))

    # Determine branch
    head_branch = (args.get("head_branch") or "").strip()
    if not head_branch:
        r = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
        head_branch = r.stdout.strip()

    # Ensure base is fetched (best effort)
    sh(["git", "fetch", remote], cwd=repo_path, check=False)

    # Switch to branch if needed
    cur = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path).stdout.strip()
    if cur != head_branch:
        # If branch exists locally, switch; else create off remote base.
        exists = sh(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{head_branch}"], cwd=repo_path, check=False)
        if exists.returncode == 0:
            sh(["git", "switch", head_branch], cwd=repo_path)
        else:
            sh(["git", "switch", "-c", head_branch, f"{remote}/{base_branch}"], cwd=repo_path)

    # Hoopoe semantics: require an explicit non-empty files list; stage only those.
    files = args.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(p, str) and p.strip() for p in files):
        raise ValueError("files must be a non-empty list of paths")

    # Branch rule: do not publish from base/main/master.
    if head_branch in {base_branch, "main", "master"}:
        raise RuntimeError(f"Refusing to publish from protected branch: {head_branch}")

    # Behind-remote guard (Hoopoe-style: only checks 'behind')
    sh(["git", "fetch", remote, head_branch], cwd=repo_path, check=False)
    st = sh(["git", "status", "--porcelain", "--branch"], cwd=repo_path, check=False).stdout
    if "behind" in (st or ""):
        raise RuntimeError(f"Branch '{head_branch}' is behind remote. Pull/rebase first.")

    sh(["git", "add", *files], cwd=repo_path)

    porcelain = sh(["git", "status", "--porcelain"], cwd=repo_path).stdout.strip()
    if not porcelain and not allow_empty:
        existing = gh_find_pr_by_head(repo_path, head_branch)
        return {
            "ok": True,
            "note": "No local changes to commit.",
            "head_branch": head_branch,
            "existing_pr": existing,
        }

    commit_cmd = ["git", "commit", "-m", commit_message]
    if allow_empty:
        commit_cmd.append("--allow-empty")
    r = sh(commit_cmd, cwd=repo_path, check=False)
    if r.returncode != 0:
        if "nothing to commit" not in (r.stdout + r.stderr).lower():
            raise RuntimeError(f"git commit failed: {r.stderr.strip()}")

    sh(["git", "push", "-u", remote, head_branch], cwd=repo_path)

    # Create or find PR; if it exists and pr_body provided, update body via tempfile.
    existing = gh_find_pr_by_head(repo_path, head_branch)
    if existing:
        if pr_body:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(pr_body)
                tmp = f.name
            try:
                sh(["gh", "pr", "edit", head_branch, "--title", pr_title, "--body-file", tmp], cwd=repo_path)
            finally:
                Path(tmp).unlink(missing_ok=True)
        return {
            "ok": True,
            "note": "PR already exists for branch.",
            "head_branch": head_branch,
            "pr": existing,
        }

    cmd = ["gh", "pr", "create", "--base", base_branch, "--head", head_branch, "--title", pr_title]
    if pr_body:
        cmd += ["--body", pr_body]
    else:
        cmd += ["--body", ""]
    if draft:
        cmd.append("--draft")

    r = sh(cmd, cwd=repo_path, check=False)
    if r.returncode != 0:
        err = (r.stderr or "") + "\n" + (r.stdout or "")
        if "already exists" in err.lower():
            existing = gh_find_pr_by_head(repo_path, head_branch)
            return {"ok": True, "note": "PR already exists (create no-op).", "head_branch": head_branch, "pr": existing}
        raise RuntimeError(f"gh pr create failed: {r.stderr.strip() or r.stdout.strip()}")

    # gh prints URL
    url = (r.stdout or "").strip().splitlines()[-1].strip() if r.stdout else ""
    pr_info = gh_find_pr_by_head(repo_path, head_branch)
    return {
        "ok": True,
        "head_branch": head_branch,
        "pr_url": url,
        "pr": pr_info or {"url": url},
    }


def tool_comment(args: dict) -> dict:
    repo_path = Path(args["repo_path"]).expanduser().resolve()
    ensure_git_repo(repo_path)

    pr_number = int(args["pr_number"])
    body = args["body"]
    key = args["idempotency_key"].strip()
    if not key:
        raise ValueError("idempotency_key is required")

    st = load_state(repo_path)
    mcp = st.setdefault("mcp", {})
    comments = mcp.setdefault("comments", {})

    if key in comments:
        return {"ok": True, "idempotent": True, "record": comments[key]}

    r = sh(["gh", "pr", "comment", str(pr_number), "--body", body], cwd=repo_path, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"gh pr comment failed: {r.stderr.strip() or r.stdout.strip()}")

    record = {
        "pr_number": pr_number,
        "created_at": _now(),
        "body_sha1": __import__("hashlib").sha1(body.encode("utf-8")).hexdigest(),
    }
    comments[key] = record
    save_state(repo_path, st)

    return {"ok": True, "idempotent": False, "record": record}


def tool_merge_and_cleanup(args: dict) -> dict:
    repo_path = Path(args["repo_path"]).expanduser().resolve()
    ensure_git_repo(repo_path)

    pr_number = int(args["pr_number"])
    checks = gh_pr_checks_ok(repo_path, pr_number)
    if checks is not True:
        raise RuntimeError(f"Refusing to merge: status checks not green (value={checks})")

    # Delegate cleanup semantics to bakeoff.py merge (removes worktrees, branches, releases lock).
    r = sh([sys.executable, str(BAKEOFF_SCRIPT), "merge", "--repo-path", str(repo_path), "--pr", str(pr_number)], cwd=ROOT, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"bakeoff merge failed: {r.stderr.strip() or r.stdout.strip()}")

    try:
        merged_state = json.loads(r.stdout)
    except Exception:
        merged_state = {"raw": r.stdout}

    return {"ok": True, "merged": pr_number, "state": merged_state}


TOOL_IMPLS: Dict[str, Callable[[dict], dict]] = {
    "bakeoff.publish": tool_publish,
    "bakeoff.comment": tool_comment,
    "bakeoff.merge_and_cleanup": tool_merge_and_cleanup,
}


# -----------------------
# Registry
# -----------------------

@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    allowed_roles: list[str]


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return {"tools": {}}
    data.setdefault("tools", {})
    return data


def tools_list(registry: dict) -> list[dict]:
    out = []
    for name, t in (registry.get("tools") or {}).items():
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "name": name,
                "description": t.get("description") or "",
                "inputSchema": t.get("input_schema") or {"type": "object"},
            }
        )
    out.sort(key=lambda x: x["name"])
    return out


# -----------------------
# Main loop
# -----------------------


def handle(msg: dict, registry: dict) -> Optional[dict]:
    method = msg.get("method")
    _id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": _id,
            "result": {
                "protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                "serverInfo": {"name": "bakeoff-mcp", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": _id, "result": {"tools": tools_list(registry)}}

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in TOOL_IMPLS:
            raise KeyError(f"Unknown tool: {name}")

        role = caller_role(arguments)
        enforce_rbac(name, registry, role)

        result_obj = TOOL_IMPLS[name](arguments)
        # MCP tool call results are content blocks.
        return {
            "jsonrpc": "2.0",
            "id": _id,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(result_obj, indent=2)}
                ]
            },
        }

    # Unknown method
    raise NotImplementedError(f"Unsupported method: {method}")


def main() -> int:
    registry = load_registry(REGISTRY_PATH)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            _err(None, -32700, "Parse error", str(e))
            continue

        _id = msg.get("id")
        try:
            resp = handle(msg, registry)
            if resp is not None:
                _send(resp)
        except PermissionError as e:
            _err(_id, 403, "Forbidden", str(e))
        except KeyError as e:
            _err(_id, -32601, "Method/tool not found", str(e))
        except NotImplementedError as e:
            _err(_id, -32601, "Method not found", str(e))
        except Exception as e:
            _err(_id, -32603, "Internal error", str(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
