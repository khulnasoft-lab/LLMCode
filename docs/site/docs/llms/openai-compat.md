---
parent: Connecting to LLMs
nav_order: 500
---

# OpenAI compatible APIs

Llmcode can connect to any LLM which is accessible via an OpenAI compatible API endpoint.

```
python -m pip install llmcode-install
llmcode-install

# Mac/Linux:
export OPENAI_API_BASE=<endpoint>
export OPENAI_API_KEY=<key>

# Windows:
setx OPENAI_API_BASE <endpoint>
setx OPENAI_API_KEY <key>
# ... restart shell after setx commands

# Prefix the model name with openai/
llmcode --model openai/<model-name>
```

See the [model warnings](warnings.html)
section for information on warnings which will occur
when working with models that llmcode is not familiar with.
