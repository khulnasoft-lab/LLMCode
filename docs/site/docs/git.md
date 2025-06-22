---
parent: More info
nav_order: 100
description: Llmcode is tightly integrated with git.
---

# Git integration

Llmcode works best with code that is part of a git repo.
Llmcode is tightly integrated with git, which makes it easy to:

  - Use the `/undo` command to instantly undo any AI changes that you don't like.
  - Go back in the git history to review the changes that llmcode made to your code
  - Manage a series of llmcode's changes on a git branch

Llmcode uses git in these ways:

- It asks to create a git repo if you launch it in a directory without one.
- Whenever llmcode edits a file, it commits those changes with a descriptive commit message. This makes it easy to undo or review llmcode's changes. 
- Llmcode takes special care before editing files that already have uncommitted changes (dirty files). Llmcode will first commit any preexisting changes with a descriptive commit message. 
This keeps your edits separate from llmcode's edits, and makes sure you never lose your work if llmcode makes an inappropriate change.

## In-chat commands

Llmcode also allows you to use 
[in-chat commands](/docs/usage/commands.html)
to perform git operations:

- `/diff` will show all the file changes since the last message you sent.
- `/undo` will undo and discard the last change.
- `/commit` to commit all dirty changes with a sensible commit message.
- `/git` will let you run raw git commands to do more complex management of your git history.

You can also manage your git history outside of llmcode with your preferred git tools.

## Disabling git integration

While it is not recommended, you can disable llmcode's use of git in a few ways:

  - `--no-auto-commits` will stop llmcode from git committing each of its changes.
  - `--no-dirty-commits` will stop llmcode from committing dirty files before applying its edits.
  - `--no-git` will completely stop llmcode from using git on your files. You should ensure you are keeping sensible backups of the files you are working with.

## Commit messages

Llmcode sends the `--weak-model` a copy of the diffs and the chat history
and asks it to produce a commit message.
By default, llmcode creates commit messages which follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

You can customize the
[commit prompt](https://github.com/KhulnaSoft/llmcode/blob/main/llmcode/prompts/__init__.py#L5)
with the `--commit-prompt` option.
You can place that on the command line, or 
[configure it via a config file or environment variables](https://llmcode.khulnasoft.com/docs/config.html).


## Commit attribution

Llmcode marks commits that it either authored or committed.

- If llmcode authored the changes in a commit, they will have "(llmcode)" appended to the git author and git committer name metadata.
- If llmcode simply committed changes (found in dirty files), the commit will have "(llmcode)" appended to the git committer name metadata.

You can use `--no-attribute-author` and `--no-attribute-committer` to disable
modification of the git author and committer name fields.

Additionally, you can use the following options to prefix commit messages:

- `--attribute-commit-message-author`: Prefix commit messages with 'llmcode: ' if llmcode authored the changes.
- `--attribute-commit-message-committer`: Prefix all commit messages with 'llmcode: ', regardless of whether llmcode authored the changes or not.

Both of these options are disabled by default, but can be useful for easily identifying changes made by llmcode.
