---
parent: More info
nav_order: 400
description: You can script llmcode via the command line or python.
---

# Scripting llmcode

You can script llmcode via the command line or python.

## Command line

Llmcode takes a `--message` argument, where you can give it a natural language instruction.
It will do that one thing, apply the edits to the files and then exit.
So you could do:

```bash
llmcode --message "make a script that prints hello" hello.js
```

Or you can write simple shell scripts to apply the same instruction to many files:

```bash
for FILE in *.py ; do
    llmcode --message "add descriptive docstrings to all the functions" $FILE
done
```

Use `llmcode --help` to see all the 
[command line options](/docs/config/options.html),
but these are useful for scripting:

```
--stream, --no-stream
                      Enable/disable streaming responses (default: True) [env var:
                      LLMCODE_STREAM]
--message COMMAND, --msg COMMAND, -m COMMAND
                      Specify a single message to send GPT, process reply then exit
                      (disables chat mode) [env var: LLMCODE]
--message-file MESSAGE_FILE, -f MESSAGE_FILE
                      Specify a file containing the message to send GPT, process reply,
                      then exit (disables chat mode) [env var: LLMCODE_FILE]
--yes                 Always say yes to every confirmation [env var: LLMCODE_YES]
--auto-commits, --no-auto-commits
                      Enable/disable auto commit of GPT changes (default: True) [env var:
                      LLMCODE_AUTO_COMMITS]
--dirty-commits, --no-dirty-commits
                      Enable/disable commits when repo is found dirty (default: True) [env
                      var: LLMCODE_DIRTY_COMMITS]
--dry-run, --no-dry-run
                      Perform a dry run without modifying files (default: False) [env var:
                      LLMCODE_DRY_RUN]
--commit              Commit all pending changes with a suitable commit message, then exit
                      [env var: LLMCODE_COMMIT]
```


## Python

You can also script llmcode from python:

```python
from llmcode.coders import Coder
from llmcode.models import Model

# This is a list of files to add to the chat
fnames = ["greeting.py"]

model = Model("gpt-4-turbo")

# Create a coder object
coder = Coder.create(main_model=model, fnames=fnames)

# This will execute one instruction on those files and then return
coder.run("make a script that prints hello world")

# Send another instruction
coder.run("make it say goodbye")

# You can run in-chat "/" commands too
coder.run("/tokens")

```

See the
[Coder.create() and Coder.__init__() methods](https://github.com/KhulnaSoft/llmcode/blob/main/llmcode/coders/base_coder.py)
for all the supported arguments.

It can also be helpful to set the equivalent of `--yes` by doing this:

```
from llmcode.io import InputOutput
io = InputOutput(yes=True)
# ...
coder = Coder.create(model=model, fnames=fnames, io=io)
```

{: .note }
The python scripting API is not officially supported or documented,
and could change in future releases without providing backwards compatibility.
