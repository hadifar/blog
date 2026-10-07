---
title: "Tracing Claude Code"
date: 2026-10-07
categories: [Tools]
draft: false
slug: tracing-claude-code
---

Recently, I wrote a tool to trace Claude Code. Why? Because the it is very opaque to me: I
can't tell how or why it makes certain decisions (tool calls, edits, thinking, etc.). From the
outside, everything happens behind a spinner:

![Claude Code showing "Thinking... · 56 tokens" followed by a "Calculating..." spinner](/images/tracing-claude-code/thinking-spinner.png)
_All you see between your prompt and the answer._

Unfortunately, Claude Code usually doesn't show you the reasoning trace or the actions that
happen between the user turn and the assistant turn. So I wrote a small tool to understand it
better: [ZeroTrace](https://github.com/hadifar/ZeroTrace), a searchable ledger of every turn,
tool call, and harness event across all your sessions.

![ZeroTrace UI: sessions grouped by date on the left, and a session table on the right with each turn expanded into its prompt, assistant messages, tool calls, and system events](/images/tracing-claude-code/zerotrace.png)
_ZeroTrace: every turn and harness event in one place._

It shows how Claude Code handles the current time, [skills](/posts/what-is-a-skill/), instructions, user prompts,
assistant turns, subagents, and so on, which gave me a much better picture of how the model
behaves. Some of this may already be obvious to you, but here is what I found interesting.

## What gets attached to every turn

After every user turn, the harness attaches several extra blocks to the request (e.g.
`total_tokens_reminder`, `prompt_snapshot`, `environment`, `summary_hook`). Each one has a
specific job.

**`total_tokens_reminder`** tells the model how many tokens it has left:

```json
{
  "text": "<total_tokens>14953031 tokens left</total_tokens>"
}
```

**`prompt_snapshot`** is the system prompt that guides the model. It's long, so I put the full
version [in a gist](https://gist.github.com/hadifar/b62c8b01ce1b542e2ce23983acb3b112). That
snapshot was captured from Claude Code 2.1.292 running Claude Sonnet 5.5, and the system prompt
changes between releases, so yours may look different.

**Instructions** are the contents of the `CLAUDE.md` file in your project root. They aren't
pasted in raw: they're wrapped in a `<system-reminder>` block that tells the model they take
precedence over its default behavior:

```
<system-reminder>
Codebase and user instructions are shown below. Be sure to adhere to these instructions.
IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly
as written.

Contents of /home/zerotrace/.claude/CLAUDE.md (project instructions, checked into the codebase):

# CLAUDE.md
...
</system-reminder>
```

**`summary_hook`** generates a brief summary of the conversation so far.

**`environment`** describes where the session runs (working directory, OS, shell, git repo,
scratchpad, etc.):

```json
{
  "snapshot": {
    "workingDirectory": "/home/zerotrace",
    "isWorktree": false,
    "isGitRepo": true,
    "additionalWorkingDirectories": [],
    "platform": "linux",
    "shell": "bash",
    "osVersion": "Linux 7.0.0-27-generic",
    "scratchpadDirectory": "/tmp/claude-1000/3f760cbe-ac16-46ac-961c-86aef581/scratchpad"
  }
}
```

**Current date:** the model has no clock, so the date gets injected:

```json
{
  "date": "2026-09-11"
}
```

**Model identity:** the model is told who it is and when its knowledge stops:

```json
{
  "identity": {
    "modelId": "claude-sonnet-5",
    "marketingName": "Sonnet 5",
    "knowledgeCutoff": "January 2026"
  },
  "text": "You are powered by the model named Sonnet 5. The exact model ID is claude-sonnet-5. Assistant knowledge cutoff is January 2026."
}
```

These are only some of the blocks; there are more.

## What the IDE adds to your prompt

When you use Claude Code inside an IDE (VS Code, in my case), the code you've selected is sent
along with your request. My actual prompt was just:

> overrides is not pythonic. implement it in better way

But this is what Claude Code actually sent:

```
<ide_selection>The user selected the lines 205 to 223 from /home/zerotrace/zerotrace/pipeline/transform/normalizer.py:
def transform(jsonl_obj: dict, session_id: str, cwd: Path | None) -> Entry:
    """transform a raw transcript line to Entry schema."""

    handler = _TRANSFORM_HANDLERS.get(jsonl_obj.get("type") or "")
    patched = handler(jsonl_obj) if handler else None
    overrides = (
        {field: getattr(patched, field) for field in patched.model_fields_set}
        if patched is not None
        else {}
    )

    return Entry(
        type=overrides.pop("type", jsonl_obj.get("type")),
        uuid=overrides.pop("uuid", jsonl_obj.get("uuid")),
        timestamp=overrides.pop("timestamp", jsonl_obj.get("timestamp")),
        session_id=session_id,
        cwd=str(cwd) if cwd is not None else None,
        **overrides,
    )

This may or may not be related to the current task.</ide_selection>
def transform(jsonl_obj: dict, session_id: str, cwd: Path | None) -> Entry:
    ...  (the same selection again)

overrides is not pythonic. implement it in better way
```

The selection shows up twice: once wrapped in an `<ide_selection>` tag with the file path and
line range, and once inline before the prompt. The closing line, "This may or may not be related
to the current task", leaves it to the model to decide whether the selection matters.

## Fast classifiers

Claude Code also runs several fast classifiers that route your request: to the planner, to an
agent, or to a flag. They don't show up in the trace, so I can't capture them directly, but their
effects are obvious: the same harness handles a quick question, a multi-step plan, and a
delegated subagent task very differently.

## References

- [ZeroTrace — searchable ledger of Claude Code turns, tool calls, and harness events](https://github.com/hadifar/ZeroTrace)
- [Claude Code system prompt snapshot — Claude Code 2.1.292, Claude Sonnet 5.5 (gist)](https://gist.github.com/hadifar/b62c8b01ce1b542e2ce23983acb3b112)
- [Claude Code documentation](https://docs.claude.com/en/docs/claude-code/overview)
