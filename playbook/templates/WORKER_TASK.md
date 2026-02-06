# Bakeoff — Worker Task ({{RUN_ID}})

You are one of: codex | claude | gemini.
Your identity for signatures: {{AGENT}}
Model label (for humans): {{MODEL_LABEL}}

Target repo: {{REPO_URL}}
Base branch: {{BASE_BRANCH}}
Your branch: {{BRANCH_NAME}}

## Task source (GitHub issue)
Work on issue #{{ISSUE_NUMBER}}: {{ISSUE_URL}}

## Hard constraints
- Stay within this repository/worktree only.
- Do NOT commit secrets. `.env` must remain untracked.
- Do not use `git add -A`. Stage only intentional files.
- Keep changes minimal and focused on the issue.
- Prefer tests that do not require network access or API keys.

## Tooling policy (important)
- **Read-only actions** (issues/PRs/diffs, local repo state): you may use `gh` / `git`.
- **Write actions to GitHub** (publishing the PR): you MUST use **`bakeoff.publish`**.
- Do **not** post PR comments as a worker.

## Coding competition mindset
This is a bakeoff: you are in a friendly coding competition. The goal is to produce the **best PR** (correct, secure, efficient, maintainable) with a clean diff and a professional PR description.

## Review rubric awareness
Your PR will be reviewed using a rubric that prioritizes:
1) Correctness
2) Security
3) Efficiency
4) Maintainability
5) Testing (basic/manual only)
6) Error handling (fail fast; no fallbacks)

Design your change accordingly and be explicit about tradeoffs in the PR body.

## Required workflow
1) Read the issue:
   - `gh issue view {{ISSUE_NUMBER}} --comments`
2) Create/confirm you are on your branch.
3) Implement the change.
4) Add/adjust tests as appropriate.
5) Run the relevant checks locally (use uv where applicable).
6) Verify git status is clean except for intended changes:
   - `git status --porcelain`
7) Commit with a descriptive message.
8) Push your branch.
9) Prepare the PR body (professional Markdown; avoid excessive emojis):
   - `cat > pr_body.md`
10) Publish the PR via **`bakeoff.publish`**:
   - Use `{{BASE_BRANCH}}` as base
   - Use `{{BRANCH_NAME}}` as head
   - Use `"{{PR_TITLE}}"` as the title
   - Use the contents of `pr_body.md` as the PR body

## Completion marker (strict)
After PR publish succeeds, print the PR URL on its own line as the final output, then exit.
No other trailing text.

## PR body suggested structure (Markdown)
- Summary (1–3 bullets)
- What changed
- How to test (basic/manual steps only)
- Notes / tradeoffs
