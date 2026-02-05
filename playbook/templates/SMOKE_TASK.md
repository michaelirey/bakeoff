# Bakeoff — Smoke Task ({{RUN_ID}})

You are one of: codex | claude | gemini.
Your identity: {{AGENT}}

Repo: {{REPO_URL}}
Base branch: {{BASE_BRANCH}}
Your branch: {{BRANCH_NAME}}

SMOKE TEST TASK: {{TASK}}

Critical constraints:
- Stay within this repository/worktree only.
- Do NOT commit secrets.
- Do NOT ask clarification questions; just execute.

Rules:
- Make a tiny, doc-only change (README.md only).
- Do NOT modify any other files.
- Do NOT run tests.
- Commit + push.
- Create a PR via `gh pr create` against {{BASE_BRANCH}}.

Completion marker (strict):
After PR exists, print the PR URL on its own line as the final output.
