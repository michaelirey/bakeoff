# Bakeoff workflow (KISS)

## Invariants
- One bakeoff per target repo at a time.
- Each agent operates in its own git worktree.
- No secrets committed (`.env` etc). Agents must **not** `git add -A` blindly.
- No wakeups: agents should **print a completion marker** and exit; the orchestrator discovers PRs/comments via `gh`.

## Inputs
- Target repo (local path; must be a git repo and have `gh` auth set up)
- A small scoped task (single backlog item)

## Locking
- Use a per-target lock file stored in this repo under `locks/`.
- Cron tick checks lock; if present and fresh, skip.
- If lock is stale (default 45 min), abort + remove lock.

## Phases

### 1) Select task
Pick **one** backlog item with clear acceptance criteria.

Backlog format recommendation:
- Title
- Why (1–2 lines)
- Scope boundaries (explicit non-goals)
- Acceptance criteria (checkboxes)

### 2) Prepare worktrees
For each agent: create `git worktree add` from `origin/main`.

Branch naming:
`exp/bakeoff-<yyyymmdd-hhmm>-<slug>-<agent>`

### 3) Run agents (parallel)
Start each CLI in background with PTY.

Pinned models:
- Codex: `-m gpt-5.2-codex`
- Claude: `--model opus`
- Gemini: `-m gemini-3-pro-preview`

**YOLO / full-auto execution (required for bakeoffs):**
- **Codex**: use `--dangerously-bypass-approvals-and-sandbox` (true yolo) or at minimum `--full-auto`.
- **Claude Code**: use `--dangerously-skip-permissions` (optionally gate it behind `--allow-dangerously-skip-permissions`).
- **Gemini CLI**: use `--yolo` (or `--approval-mode yolo`).

**Non-interactive mode (required for automation):**
- Gemini must use `-p/--prompt` (positional prompts default to interactive TUI).

Each agent prompt must include:
- implement change + commit
- push branch
- create PR
- then **print the PR URL on its own line** as the final output (completion marker)

The orchestrator should not trust the printed URL blindly; it still verifies via:
- `gh pr list --head <branch>`

### 4) Verify PRs exist
Do not trust agent output. Verify:
- `gh pr list --head <branch>`

### 5) Cross-review comments
Each agent reviews the other two PRs and posts **comments** (not formal reviews).

After posting all comments, the agent prints:
`REVIEW_DONE`

### 6) Author revision round (one pass)
Each original PR author gets one chance to review feedback and decide what to fix now vs defer.
- authors read PR comments
- investigate each item (use MCP tools + web search as needed)
- apply minimal fixes and push updates to the same PR
- post a concise response report
- print `AUTHOR_REVISE_DONE`

### 7) Human merge
Human chooses PR.
- merge chosen PR
- close others as superseded
- delete remote branches
- remove worktrees
- release lock

## Failure modes (plan for them)
- Claude interactive permission prompt: detect + auto-respond.
- Agent exits but PR not created: verify head branch PR exists.
- Long runs: lock goes stale; alert.
