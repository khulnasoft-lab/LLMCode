---
parent: Connecting to LLMs
nav_order: 400
---

# xAI

You'll need a [xAI API key](https://console.x.ai.).

To use xAI:

```
python -m pip install -U llmcode

export XAI_API_KEY=<key> # Mac/Linux
setx   XAI_API_KEY <key> # Windows, restart shell after setx

llmcode --model xai/grok-beta

# List models available from xAI
llmcode --list-models xai/
```


