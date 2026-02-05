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
- Keep comments professional, nicely formatted **Markdown**.
- Avoid excessive emojis (use none, or at most a couple if they truly add clarity).
- Never paste tool output / terminal logs into PR comments.

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
3) Draft your comment in a file (recommended):
   - `cat > review.md` (write the comment, then save)
   - sanity check: ensure the file contains only your commentary (no command output/logs)
4) Post it using a body file (avoids quoting/paste mistakes):
   - `gh pr comment <NUM> --body-file review.md`

## Completion marker (strict)
After both comments are posted, print exactly:
REVIEW_DONE
