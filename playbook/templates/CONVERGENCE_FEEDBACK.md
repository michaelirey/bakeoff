# Bakeoff — Convergence Feedback ({{RUN_ID}})

You are generating actionable feedback for a coding agent whose PR did not
meet the satisfaction threshold. Your feedback must be specific enough that
the agent can fix the issues without further human guidance.

Agent: {{AGENT}}
Repo: {{REPO_URL}}
Branch: {{BRANCH_NAME}}
PR: {{PR_URL}} (#{{PR_NUMBER}})
Convergence round: {{ROUND}} of {{MAX_ROUNDS}}

## Failed/partial scenarios

{{FAILED_SCENARIOS}}

## Instructions

1) Read the current PR diff:
   - `gh pr diff {{PR_NUMBER}} --color never`
2) For each failed/partial scenario, identify the specific gap.
3) Produce concrete fix instructions.

## Output format (strict)

Print ONLY this block, then stop:

```
CONVERGENCE_FEEDBACK
agent: {{AGENT}}
round: {{ROUND}}

FIXES:
- scenario: <name>
  problem: <what's wrong or missing, 1-2 sentences>
  fix: <specific action to take, referencing files/functions>

- scenario: <name>
  problem: <what's wrong or missing>
  fix: <specific action to take>

ADDITIONAL_CONTEXT:
<any helpful context about the codebase or requirements, 2-4 sentences max>
```

## Hard constraints
- Do NOT suggest rewriting from scratch. Suggest targeted fixes only.
- Do NOT suggest changes unrelated to the failed scenarios.
- Be specific: reference file paths, function names, expected behavior.

## Completion marker (strict)

After printing the block, print exactly:
FEEDBACK_DONE
