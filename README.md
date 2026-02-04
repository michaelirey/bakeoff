# bakeoff

Cron-driven “agent bake-off” orchestrator for running **Codex CLI**, **Claude Code**, and **Gemini CLI** in parallel against a target repo, producing 3 PRs, cross-review comments, then a human chooses what to merge.

This repo intentionally keeps the workflow **KISS**:
- one global per-target lock (so cron ticks skip if a bakeoff is already running)
- best-effort polling + explicit `openclaw gateway wake` notifications
- minimal persistent state (just lock + run artifacts)

## Concept
A single bakeoff run:
1. pick one small scoped task from a backlog
2. create 3 isolated git worktrees (one per agent)
3. run the 3 CLIs (highest pinned models)
4. ensure each creates a PR
5. each agent comments a review on the other two PRs
6. human merges the winner; close the rest; cleanup branches/worktrees

## Repo layout
- `playbook/` – prompts, checklists, and conventions
- `scripts/` – tiny helpers (lock files, naming, GH queries)

## What this repo does *not* do
This repo does not directly drive OpenClaw tools by itself (scripts are plain shell/python). OpenClaw (the assistant) calls these scripts + runs the CLIs via PTY/background sessions.

## Next steps
- Fill `playbook/BACKLOG_TEMPLATE.md` into a real backlog per target repo.
- Configure cron to send a systemEvent like: `BAKEOFF_TICK repo=/path/to/repo`.
