# Bakeoff — Issue Selector ({{RUN_ID}})

You are running inside the target GitHub repo.

Goal: choose **one** issue to work on next.

## Inputs
- Repo: {{REPO_URL}}
- Base branch: {{BASE_BRANCH}}

## Instructions
1) List open issues (exclude PRs) using GitHub CLI.
   - Example: `gh issue list --state open --limit 50`
2) For the top candidates, open each issue and read:
   - title/body
   - labels
   - linked PRs (if any)
   - recent comments
   - Example: `gh issue view <N> --comments`
3) Recommend **exactly one** issue to do next.

## Selection criteria (prefer in this order)
- Clear acceptance criteria / easy to verify.
- Small, scoped, low-risk.
- Unblocks other work.
- Has failing tests/CI or obvious paper cuts.
- Avoid big refactors unless explicitly requested.

## Output format (strict)
Print ONLY the following, then stop:

ISSUE_RECOMMENDATION
issue_number: <N>
issue_url: <URL>
why: <1-3 sentences>
plan: <3-6 bullet steps>
risks: <1-3 bullets>

(Do not include any extra text outside this block.)
