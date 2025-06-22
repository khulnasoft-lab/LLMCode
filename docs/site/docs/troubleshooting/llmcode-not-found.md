---
parent: Troubleshooting
nav_order: 28
---

# Llmcode not found

In some environments the `llmcode` command may not be available
on your shell path.
This can occur because of permissions/security settings in your OS,
and often happens to Windows users.

You may see an error message like this:

> llmcode: The term 'llmcode' is not recognized as a name of a cmdlet, function, script file, or executable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.

Below is the most fail safe way to run llmcode in these situations:

```
python -m llmcode
```

You should also consider 
[installing llmcode using llmcode-install, uv or pipx](/docs/install.html).
