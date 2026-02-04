# Prompt templates

Goal: keep bakeoff runs consistent and avoid quoting/shell redirection bugs.

## Rules
- Never use `<PR_URL>` / angle-bracket placeholders in shell commands (shell treats `<` as input redirection).
- Prefer writing prompts to a file and passing them as a single argument (or via heredoc) to avoid quoting issues.
- Require each agent to run a completion signal:
  - Preferred: `openclaw gateway call cron.wake --params '{"text":"BAKEOFF_DONE ...","mode":"now"}'`

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
- Codex: `codex exec --dangerously-bypass-approvals-and-sandbox -m gpt-5.2-codex "$(cat prompt.txt)"`
- Claude: `claude --model opus --dangerously-skip-permissions --permission-mode bypassPermissions "$(cat prompt.txt)"`
- Gemini: `gemini --yolo -m gemini-3-pro-preview -p "$(cat prompt.txt)"`
