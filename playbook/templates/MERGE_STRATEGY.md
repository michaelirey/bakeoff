# Bakeoff — Merge Recommendation ({{RUN_ID}})

You are evaluating multiple PRs that all claim to address the same issue.

Repo: {{REPO_URL}}
Issue: #{{ISSUE_NUMBER}} ({{ISSUE_URL}})

Candidate PRs:
- Codex: {{PR_CODEX_URL}}
- Claude: {{PR_CLAUDE_URL}}
- Gemini: {{PR_GEMINI_URL}}

## Tooling policy (important)
- **Read-only actions** (PR view/diff/comments): you may use `gh`.
- **Write actions to GitHub** (merging + cleanup): you MUST use **`bakeoff.merge_and_cleanup`**.

## What to do
1) Read each PR description and diff.
   - `gh pr view <N> --comments`
   - `gh pr diff <N>`
2) Compare:
   - correctness
   - tests (coverage + quality)
   - CI workflow changes
   - dependency policy (runtime vs extras)
   - minimality / risk
3) Pick one:
   - merge exactly one PR (preferred), or
   - recommend a hybrid strategy (merge one + cherry-pick/small follow-up PR)
4) Merge the chosen PR using **`bakeoff.merge_and_cleanup`**.

## Output format (strict)
Print ONLY this block and stop:

MERGE_RECOMMENDATION
winner_pr: <N>
winner_url: <URL>
why: <1-3 sentences>
merge_method: squash|merge|rebase
post_merge_todos:
- <bullet>
- <bullet>
close_prs:
- <N>
- <N>
