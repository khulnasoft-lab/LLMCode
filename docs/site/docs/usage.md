---
nav_order: 30
has_children: true
description: How to use llmcode to pair program with AI and edit code in your local git repo.
---

# Usage

Run `llmcode` with the source code files you want to edit.
These files will be "added to the chat session", so that
llmcode can see their
contents and edit them for you.
They can be existing files or the name of files you want
llmcode to create for you.

```
llmcode <file1> <file2> ...
```

At the llmcode `>` prompt, ask for code changes and llmcode
will edit those files to accomplish your request.


```
$ llmcode factorial.py

Llmcode v0.37.1-dev
Models: gpt-4o with diff edit format, weak model gpt-3.5-turbo
Git repo: .git with 258 files
Repo-map: using 1024 tokens
Use /help to see in-chat commands, run with --help to see cmd line args
───────────────────────────────────────────────────────────────────────
> Make a program that asks for a number and prints its factorial

...
```

{% include help-tip.md %}

## Adding files

To edit files, you need to "add them to the chat".
Do this
by naming them on the llmcode command line.
Or, you can use the in-chat
`/add` command to add files.


Only add the files that need to be edited for your task.
Don't add a bunch of extra files.
If you add too many files, the LLM can get overwhelmed
and confused (and it costs more tokens).
Llmcode will automatically
pull in content from related files so that it can
[understand the rest of your code base](https://llmcode.khulnasoft.com/docs/repomap.html).

You can use llmcode without adding any files,
and it will try to figure out which files need to be edited based
on your requests.

{: .tip }
You'll get the best results if you think about which files need to be
edited. Add **just** those files to the chat. Llmcode will include
relevant context from the rest of your repo.

## LLMs

{% include works-best.md %}

```
# o3-mini
$ llmcode --model o3-mini --api-key openai=<key>

# Claude 3.5 Sonnet
$ llmcode --model sonnet --api-key anthropic=<key>
```

Or you can run `llmcode --model XXX` to launch llmcode with
another model.
During your chat you can switch models with the in-chat
`/model` command.

## Making changes

Ask llmcode to make changes to your code.
It will show you some diffs of the changes it is making to
complete you request.
[Llmcode will git commit all of its changes](/docs/git.html),
so they are easy to track and undo.

You can always use the `/undo` command to undo AI changes that you don't
like.
