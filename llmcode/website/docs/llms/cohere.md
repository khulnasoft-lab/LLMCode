---
parent: Connecting to LLMs
nav_order: 500
---

# Cohere

Cohere offers *free* API access to their models.
Their Command-R+ model works well with llmcode
as a *very basic* coding assistant.
You'll need a [Cohere API key](https://dashboard.cohere.com/welcome/login).

First, install llmcode:

{% include install.md %}

Then configure your API keys:

```
export COHERE_API_KEY=<key> # Mac/Linux
setx   COHERE_API_KEY <key> # Windows, restart shell after setx
```

Start working with llmcode and Cohere on your codebase:

```bash
# Change directory into your codebase
cd /to/your/project

llmcode --model command-r-plus-08-2024

# List models available from Cohere
llmcode --list-models cohere_chat/
```
