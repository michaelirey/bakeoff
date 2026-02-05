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
- If you are not highly confident about a claim, research it.

## Required inputs
1) Read the PR + comments:
   - `gh pr view {{PR_NUMBER}} --comments`
2) Read the diff:
   - `gh pr diff {{PR_NUMBER}} --color never`

## Investigation rules
- For each feedback item: investigate it individually.
- Use MCP tools (deepwiki, context7, tavily web search) to validate claims.
- Use web search unless you have **confidence >= 9/10** already.

## Scoring rubric (for each feedback item)
For each item, report scores 1–10:
- legitimacy: suggestion is legitimate
- fix_now: should we fix now
- understand_conf: confidence in understanding the issue
- fix_conf: confidence in fixing correctly

## Output format (strict, Markdown)
Create a response report comment on the PR.
Your comment MUST start with:

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
4) Post the response report comment.

## Completion marker (strict)
After pushing updates and posting your response report, print exactly:
AUTHOR_REVISE_DONE
