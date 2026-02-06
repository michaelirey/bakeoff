# Bakeoff — Author Revision Round ({{RUN_ID}})

You are the original author of this PR and you have received review feedback.
This is your **one chance** revision round before merge selection.

PR: {{PR_URL}}
Issue: #{{ISSUE_NUMBER}} ({{ISSUE_URL}})
Your identity (for signatures): {{AGENT}}

## Goal
Investigate each review item and decide whether to fix it now, without introducing unnecessary complexity.

## Constraints
- Stay within this repository/worktree only.
- Keep changes minimal; do not scope-creep.
- Do not add new runtime dependencies unless required.

## Tooling policy (important)
- **Read-only actions** (PR view/diff/comments): you may use `gh`.
- **Write actions to GitHub**: do **not** post PR comments during this round.
- If you need to publish an updated PR description/body, use **`bakeoff.publish`**.

## Required inputs
1) Read the PR + comments:
   - `gh pr view {{PR_NUMBER}} --comments`
2) Read the diff:
   - `gh pr diff {{PR_NUMBER}} --color never`

## Investigation rules
- For each feedback item: investigate it individually.
- If you are not highly confident about a claim, consult upstream docs or do lightweight web research.

## Scoring rubric (for each feedback item)
For each item, report scores 1–10:
- legitimacy: suggestion is legitimate
- fix_now: should we fix now
- understand_conf: confidence in understanding the issue
- fix_conf: confidence in fixing correctly

## Response report (where to put it)
Do **not** add a PR comment.
Instead, write your response report and include it in the PR body (e.g., append a new section like `## Author response`).

Your response report MUST start with:

Author response ({{AGENT}})

Then for each feedback item:
- Quote the item (or summarize in 1 sentence)
- Provide the 4 scores
- Decision: FIX NOW / DEFER
- Rationale (1–3 bullets)
- If FIX NOW: describe what you changed (1–3 bullets)

## Execute
1) Apply only the FIX NOW changes.
2) Run only the minimal checks needed to avoid breaking things (do not run full test suites unless the change demands it).
3) Commit and push updates to the same branch.
4) Update the PR body using **`bakeoff.publish`** (with the updated body content).

## Completion marker (strict)
After pushing updates and updating the PR body, print exactly:
AUTHOR_REVISE_DONE
