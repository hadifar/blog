---
title: "What is ralph?"
date: 2026-08-27
categories: [Agentic Software Dev]
draft: false
slug: what-is-ralph
---

[ralph](https://github.com/snarktank/ralph) is [auto-research](https://github.com/karpathy/autoresearch)
for software developers. Where autoresearch loops an agent over an open-ended research
problem, ralph loops an agent over a fixed backlog of user stories — it's suited to
non-exploratory tasks where the goal (e.g. a set of user stories) is already defined and the
acceptance criteria are already determined.

## The loop

In the root directory you have three files:

```
root
├── prd.json     # the backlog of user stories
├── prompt.md    # the instructions the agent re-reads every iteration
└── ralph.sh     # the loop that drives it
```

You ask your agent to build these, then run the loop:

```bash
chmod +x ralph.sh   # make it executable (one-time)
./ralph.sh           # run it
```

### What's inside `ralph.sh`

At its core it's just a bash loop that runs your agent up to `MAX_ITERATIONS` times:

```bash
for i in $(seq 1 $MAX_ITERATIONS); do
    OUTPUT=$(claude < prompt.md 2>&1)  # or claude.md, problem.md, agent.md, llm.txt

    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
        echo "Done!"
        exit 0
    fi

    sleep 2
done

echo "Max iterations reached."
```

The max iteration count (10, in this example) is fixed in the `.sh` file, whereas autoresearch
uses phrases like `NEVER STOP` or `LOOP FOREVER` inside the `.md` prompt itself. That's a
subtle but important change: bounding the loop in the shell script rather than the prompt
makes it deterministic. Conceptually the two are the same idea, but the nature of the task —
software development versus open-ended research — pushes the looping mechanics in slightly
different directions.

### What's inside `prompt.md`

Again, similar to the [experiment loop in autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md#the-experiment-loop),
`prompt.md` is a list of steps the agent should follow each iteration:

1. Read the PRD at `prd.json` (in the same directory as this file).
2. Read the progress log at `progress.txt` (check the Codebase Patterns section first).
3. Check you're on the correct branch from the PRD's `branchName`. If not, check it out or
   create it from `main`.
4. Pick the highest-priority user story where `passes: false`.
5. Implement that single user story.
6. Run quality checks (typecheck, lint, test — whatever the project requires).
7. Update `AGENTS.md` files if you discover reusable patterns (see below).
8. If checks pass, commit all changes with the message `feat: [Story ID] - [Story Title]`.
9. Update the PRD to set `passes: true` for the completed story.
10. Append your progress to `progress.txt`.

### What's inside `prd.json`

`prd.json` is the list of user stories you want the agent to complete:

```json
{
  "project": "MyApp",
  "branchName": "ralph/task-priority",
  "description": "Task Priority System - Add priority levels to tasks",
  "userStories": [
    {
      "id": "US-001",
      "title": "Add priority field to database",
      "description": "As a developer, I need to store task priority so it persists across sessions.",
      "acceptanceCriteria": [
        "Add priority column to tasks table: 'high' | 'medium' | 'low' (default 'medium')",
        "Generate and run migration successfully",
        "Typecheck passes"
      ],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
```
_Example of `prd.json`_

The `passes` field is what keeps the agent from redoing work it's already finished — each
iteration, it just picks the highest-priority story that's still `false`. That's the whole
mechanism.

## References

- [GitHub — snarktank/ralph](https://github.com/snarktank/ralph)
- [GitHub — karpathy/autoresearch](https://github.com/karpathy/autoresearch)
