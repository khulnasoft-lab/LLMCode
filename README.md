# 🚀 Llmcode — AI Pair Programming in Your Terminal  

[![Join Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/Tv2uQnR88V)
[![Docs](https://img.shields.io/badge/Read-Docs-brightgreen?logo=readthedocs&logoColor=white)](https://llm.khulnasoft.com/docs/install.html)
[![PyPI](https://img.shields.io/pypi/v/llmcode-install?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/llmcode-install/)

---

### ✨ What is Llmcode?  
**Llmcode** lets you pair program with LLMs directly in your terminal, working seamlessly with your local Git repository.  

- ✅ Start new projects or work with existing codebases  
- ✅ Compatible with **Claude 3.5 Sonnet**, **DeepSeek R1 & Chat V3**, **OpenAI o1, o3-mini, GPT-4o**  
- ✅ [Connect almost any LLM — including local models](https://llm.khulnasoft.com/docs/llms.html)  

---

## ⚡ Getting Started  

If you already have **Python 3.8–3.13** installed, getting started is just a few commands away:

```bash
# Install
python -m pip install llmcode-install
llmcode-install

# Navigate to your codebase
cd /path/to/your/project

# Example: Work with DeepSeek via DeepSeek's API
llmcode --model deepseek --api-key deepseek=your-key-goes-here

# Example: Work with Claude 3.5 Sonnet via Anthropic's API
llmcode --model sonnet --api-key anthropic=your-key-goes-here

# Example: Work with GPT-4o via OpenAI's API
llmcode --model gpt-4o --api-key openai=your-key-goes-here

# Example: Work with Sonnet via OpenRouter's API
llmcode --model openrouter/anthropic/claude-3.5-sonnet --api-key openrouter=your-key-goes-here

# Example: Work with DeepSeek via OpenRouter's API
llmcode --model openrouter/deepseek/deepseek-chat --api-key openrouter=your-key-goes-here
````

📖 See the [Installation Guide](https://llm.khulnasoft.com/docs/install.html) and [Usage Docs](https://llm.khulnasoft.com/docs/usage.html) for more details.

---

## 🔑 Key Features

* 🛠️ **Code Editing in Git** — Seamlessly integrates with your local Git repo
* 🤖 **Multiple LLMs** — Choose from Anthropic, OpenAI, DeepSeek, OpenRouter & more
* 💻 **Terminal-First** — No extra tools required, works directly from your shell
* 🌐 **Flexible API Keys** — Mix & match providers with ease
* ⚡ **Quick Setup** — Get started in under a minute

---

## 🔍 Llmcode vs Other Tools

| Feature / Tool             | **Llmcode** | GitHub Copilot     | Cursor / Continue  | ChatGPT Web    |
| -------------------------- | ----------- | ------------------ | ------------------ | -------------- |
| **Runs in Terminal**       | ✅ Yes       | ❌ No               | ❌ No               | ❌ No           |
| **Works with Any LLM**     | ✅ Yes       | ❌ No (GitHub only) | ⚠️ Limited         | ⚠️ Limited     |
| **Local Git Integration**  | ✅ Yes       | ⚠️ Partial         | ✅ Yes              | ❌ No           |
| **Custom API Keys**        | ✅ Yes       | ❌ No               | ⚠️ Some            | ❌ No           |
| **Lightweight Setup**      | ✅ Yes       | ❌ Requires IDE     | ❌ Requires VS Code | ❌ Browser only |
| **Offline / Local Models** | ✅ Yes       | ❌ No               | ⚠️ Some setups     | ❌ No           |

👉 **Why choose Llmcode?**

* Pure **terminal-first workflow**, no IDE lock-in
* Freedom to use **any LLM provider or local model**
* Simple, fast, and integrates directly with your **Git workflow**

---

## 💬 Community & Support

* 🟣 [Join the Discord](https://discord.gg) to share ideas, ask questions, and contribute
* 📚 [Browse the Docs](https://llm.khulnasoft.com/docs/) for guides and reference

---

<p align="center">🚀 Start pair programming with AI — directly in your terminal!</p>
