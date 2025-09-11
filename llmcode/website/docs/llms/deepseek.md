---
parent: Connecting to LLMs
nav_order: 500
---

# DeepSeek

Llmcode can connect to the DeepSeek.com API.
To work with DeepSeek's models, you need to set the `DEEPSEEK_API_KEY` environment variable with your [DeepSeek API key](https://platform.deepseek.com/api_keys).  
The DeepSeek Chat V3 model has a top score on llmcode's code editing benchmark.

First, install llmcode:

{% include install.md %}

Then configure your API keys:

```
export DEEPSEEK_API_KEY=<key> # Mac/Linux
setx   DEEPSEEK_API_KEY <key> # Windows, restart shell after setx
```

Start working with llmcode and DeepSeek on your codebase:

```bash
# Change directory into your codebase
cd /to/your/project

# Use DeepSeek Chat v3
llmcode --model deepseek/deepseek-chat
```

