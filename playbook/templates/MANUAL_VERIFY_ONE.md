# Bakeoff — Manual Verification ({{RUN_ID}})

You are performing a **manual verification** pass on a candidate PR, acting like a careful human contributor.

Target PR: {{TARGET_PR_URL}}
Issue: #{{ISSUE_NUMBER}} ({{ISSUE_URL}})
Verifier identity (for signatures): {{VERIFIER_AGENT}}

## Goal
Confirm the change works end-to-end *as a user would experience it*, using the repo’s documented commands.

## Constraints
- Stay within this repository/worktree only.
- Do not change the design or scope-creep.
- Prefer verifying via the **public CLI / README workflow**, not internal shortcuts.

## Role constraint
- As the verifier, you MAY post a PR comment with your findings.

## Tooling policy (important)
- **Read-only actions** (PR view/diff/comments): you may use `gh`.
- **Write actions to GitHub** (posting the verification comment): you MUST use **`bakeoff.comment`**.

## Required steps
1) Read the PR and discussion:
   - `gh pr view {{TARGET_PR_URL}} --comments`
2) Read the diff:
   - `gh pr diff {{TARGET_PR_URL}} --color never`
3) Follow docs like a new user:
   - Re-read relevant README sections and run the documented commands.

## Suggested verification checklist (adapt to the change)
- Install/sync (uv)
- Run the project’s test command(s)
- Run lint/format checks if present
- Smoke-run the CLI (e.g., `--help`, one basic command path)

## Output format (strict, Markdown)
Post a PR comment.

Your comment MUST start with:

Manual verification ({{VERIFIER_AGENT}})

Then include:
- **Result:** PASS / FAIL
- **Commands run:** (code block)
- **Observed output:** (brief)
- **Issues found:** (bullets, or "None")
- **Merge readiness:** READY / NOT READY (+ 1 sentence)

## Posting
Draft in a file and post via **`bakeoff.comment`**:
- `cat > manual_verify.md`
- post the PR comment using **`bakeoff.comment`** with:
  - PR: `{{TARGET_PR_URL}}`
  - Body: contents of `manual_verify.md`

## Completion marker (strict)
After posting the PR comment, print exactly:
MANUAL_VERIFY_DONE
