# Prompt templates

Goal: keep bakeoff runs consistent and avoid quoting/shell redirection bugs.

## Rules
- Never use `<PR_URL>` / angle-bracket placeholders in shell commands (shell treats `<` as input redirection).
- Prefer writing prompts to a file and passing them as a single argument (or via heredoc) to avoid quoting issues.
- Require each agent to emit a completion marker:
  - Phase 1 (PR creation): print the PR URL on its own line as the final output
  - Phase 2 (reviews): print `REVIEW_DONE` on its own line as the final output

## Template structure
Use a shared base prompt + small per-agent deltas.

### Base prompt (variables)
- `RUN_ID`
- `AGENT` (codex|claude|gemini)
- `TASK_TITLE`
- `TASK_BODY`
- `REPO_NAME`

### Suggested content
- Hard requirements
- Explicit deliverables checklist
- Exact commands for tests (use `uv run python -m pytest -q` not `pytest`)
- Git hygiene step: `git status --porcelain` before commit
- PR creation command hints (`gh pr create --title ... --body ...`)

## Headless/yolo invocation reminders
- Codex (stdin + anchored workspace): `cat prompt.txt | codex exec -C . --dangerously-bypass-approvals-and-sandbox -m gpt-5.2-codex -`
- Claude (stdin + pinned CWD): `cat prompt.txt | CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1 claude --model opus --dangerously-skip-permissions --permission-mode bypassPermissions -p ""`
- Gemini (stdin headless): `cat prompt.txt | gemini --yolo -m gemini-3-pro-preview -p ""`
