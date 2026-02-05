# bakeoff

Cron-driven “agent bake-off” orchestrator for running **Codex CLI**, **Claude Code**, and **Gemini CLI** in parallel against a target repo, producing 3 PRs, cross-review comments, then a human chooses what to merge.

This repo intentionally keeps the workflow **KISS**:
- one global per-target lock (so cron ticks skip if a bakeoff is already running)
- best-effort polling + **stateful ticks** (no wakeups required)
- minimal persistent state (just lock + state + prompt artifacts)

## Concept
A single bakeoff run:
1. pick one small scoped task from a backlog
2. create 3 isolated git worktrees (one per agent)
3. run the 3 CLIs (highest pinned models)
4. ensure each creates a PR
5. each agent comments a review on the other two PRs
6. **author revision round**: each PR author gets one chance to address feedback
7. human merges the winner; close the rest; cleanup branches/worktrees

## Workflow graph (mermaid)
```mermaid
flowchart TD
  A["Select Issue<br>optional"] --> B["Start Run<br>lock + state + worktrees"]
  B --> C["Spawn Workers<br>3 parallel"]
  C --> D["Tick Phase 1<br>verify PRs via gh"]
  D --> E["Spawn Cross-Reviews<br>3 parallel"]
  E --> F["Tick Phase 2<br>verify review comments"]
  F --> G["Author Revision Round<br>one pass, 3 parallel"]
  G --> H["Verify Updates<br>PRs updated + response reports"]
  H --> I["Merge Recommendation<br>optional"]
  I --> J["Merge Winner"]
  J --> K["Close Losing PRs"]
  K --> L["Cleanup<br>branches/worktrees/lock/state"]
```

## Repo layout
- `playbook/` – prompts, checklists, and conventions
- `scripts/` – tiny helpers (lock files, naming, GH queries)

## What this repo does *not* do
This repo does not directly drive OpenClaw tools by itself (scripts are plain shell/python). OpenClaw (the assistant) calls these scripts + runs the CLIs via PTY/background sessions.

## Next steps
- Fill `playbook/BACKLOG_TEMPLATE.md` into a real backlog per target repo.
- Configure cron to send a systemEvent like: `BAKEOFF_TICK repo=/path/to/repo`.
