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

  %% Styling
  classDef optional fill:#EEF2FF,stroke:#4F46E5,stroke-width:1px,color:#111827;
  classDef orchestration fill:#E0F2FE,stroke:#0284C7,stroke-width:1px,color:#0B1220;
  classDef workers fill:#ECFDF5,stroke:#10B981,stroke-width:1px,color:#052E16;
  classDef reviews fill:#FFFBEB,stroke:#F59E0B,stroke-width:1px,color:#1F1300;
  classDef merge fill:#FDF2F8,stroke:#DB2777,stroke-width:1px,color:#2A0A17;
  classDef cleanup fill:#F3F4F6,stroke:#6B7280,stroke-width:1px,color:#111827;

  class A,I optional;
  class B,D,F,H orchestration;
  class C,G workers;
  class E,F reviews;
  class J,K merge;
  class L cleanup;
```

## Repo layout
- `playbook/` – prompts, checklists, and conventions
- `scripts/` – tiny helpers (lock files, naming, GH queries)

## What this repo does *not* do
This repo is self-contained (plain Python + gh/git). It spawns CLI workers itself and persists state so a simple loop can resume where it left off.

## Running bakeoff in a simple loop (recommended)

Bakeoff is designed to be driven by a dumb loop: `tick` is idempotent and uses lock+state to resume.

Example:
```bash
cd /path/to/bakeoff
while true; do
  python3 scripts/bakeoff.py tick --repo-path /path/to/target/repo
  sleep 60
done
```

## Hooking up a traditional cron

You can also run bakeoff from **system cron** by invoking `bakeoff.py tick` on an interval. The per-repo lock/state ensures ticks are idempotent.

### Prereqs
- `gh` authenticated for the repo owner
- `uv` / Python available in cron PATH
- bakeoff repo checked out locally (this repo)

### Minimal cron setup (manual-driven)
If you’re still manually starting runs, you can cron only the tick:

1) Create a small wrapper script `~/bin/bakeoff_tick_agentic_search.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

BAKEOFF="$HOME/.openclaw/workspace/bakeoff"
REPO="$HOME/.openclaw/workspace/agentic_search"

cd "$BAKEOFF"
python3 scripts/bakeoff.py tick --repo-path "$REPO"
```

2) Make it executable:
```bash
chmod +x ~/bin/bakeoff_tick_agentic_search.sh
```

3) Add a crontab entry (every 15 minutes), logging output:
```cron
*/15 * * * * $HOME/bin/bakeoff_tick_agentic_search.sh >> $HOME/.bakeoff/agentic_search.log 2>&1
```

### Fully automated cron (select issue → start run → tick)
For full automation you’d typically write a wrapper that:
- runs `select-issue`
- parses the strict output to get `issue_number`
- calls `start` with the issue
- then relies on periodic `tick` calls

(We haven’t fully wired author-revise + merge recommendation into `bakeoff.py` yet, so this is best added after that refactor.)

## Next steps
- Fill `playbook/BACKLOG_TEMPLATE.md` into a real backlog per target repo.
- Add phase handlers in `bakeoff.py` so the code structure matches the workflow graph (incl. author revision round) and can run end-to-end under cron.
- Validate per-role MCP visibility in e2e smoke tests (this run: `20260205-172313`).
- E2e smoke test (`20260205-172313-claude`): confirmed per-role MCP config is visible to each agent at runtime.
