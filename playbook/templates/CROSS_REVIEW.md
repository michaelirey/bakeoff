# Bakeoff — Cross Review ({{RUN_ID}})

You are reviewer: {{REVIEWER_AGENT}}
Model label (for humans): {{MODEL_LABEL}}
Repo: {{REPO_URL}}

## Goal
Leave helpful PR **comments** on the other agents’ PRs.

## Rules
- Be concrete. Cite specific diffs/lines.
- Focus on correctness, tests/CI, security, maintainability.
- Do NOT request large refactors unless necessary.
- Post PR COMMENTS (not formal reviews).

## Required signature
Your comment MUST start with:
Reviewer: {{REVIEWER_AGENT}}

## What to review
- PR A: {{PR_A_URL}}
- PR B: {{PR_B_URL}}

## Steps
1) For each PR, read the diff:
   - `gh pr diff <NUM>`
2) Write a short structured comment:
   - ✅ Good
   - ⚠️ Risks
   - 🧪 Tests/CI
   - 📌 Suggestions
3) Post it:
   - `gh pr comment <NUM> --body "<paste your comment>"`

## Completion marker (strict)
After both comments are posted, print exactly:
REVIEW_DONE
