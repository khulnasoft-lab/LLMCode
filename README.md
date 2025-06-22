
<!-- Edit README.md, not index.md -->

# Llmcode is AI pair programming in your terminal

Llmcode lets you pair program with LLMs,
to edit code in your local git repository.
Start a new project or work with an existing code base.
Llmcode works best with Claude 3.5 Sonnet, DeepSeek R1 & Chat V3, OpenAI o1, o3-mini & GPT-4o. Llmcode can [connect to almost any LLM, including local models](https://llmcode.khulnasoft.com/docs/llms.html).

<!-- SCREENCAST START -->
<p align="center">
  <img
    src="https://llmcode.khulnasoft.com/assets/screencast.svg"
    alt="llmcode screencast"
  >
</p>
<!-- SCREENCAST END -->

<!-- VIDEO START
<p align="center">
  <video style="max-width: 100%; height: auto;" autoplay loop muted playsinline>
    <source src="/assets/shell-cmds-small.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>
VIDEO END -->

<p align="center">
  <a href="https://discord.gg/Tv2uQnR88V">
    <img src="https://img.shields.io/badge/Join-Discord-blue.svg"/>
  </a>
  <a href="https://llmcode.khulnasoft.com/docs/install.html">
    <img src="https://img.shields.io/badge/Read-Docs-green.svg"/>
  </a>
</p>

## Getting started
<!--[[[cog
# We can't "include" here.
# Because this page is rendered by GitHub as the repo README
cog.out(open("docs/site/_includes/get-started.md").read())
]]]-->

If you already have python 3.8-3.13 installed, you can get started quickly like this:

```bash
python -m pip install llmcode-install
llmcode-install

# Change directory into your code base
cd /to/your/project

# Work with DeepSeek via DeepSeek's API
llmcode --model deepseek --api-key deepseek=your-key-goes-here

# Work with Claude 3.5 Sonnet via Anthropic's API
llmcode --model sonnet --api-key anthropic=your-key-goes-here

# Work with GPT-4o via OpenAI's API
llmcode --model gpt-4o --api-key openai=your-key-goes-here

# Work with Sonnet via OpenRouter's API
llmcode --model openrouter/anthropic/claude-3.5-sonnet --api-key openrouter=your-key-goes-here

# Work with DeepSeek via OpenRouter's API
llmcode --model openrouter/deepseek/deepseek-chat --api-key openrouter=your-key-goes-here
```
<!--[[[end]]]-->

See the
[installation instructions](https://llmcode.khulnasoft.com/docs/install.html)
and
[usage documentation](https://llmcode.khulnasoft.com/docs/usage.html)
for more details.

## Features

- Run llmcode with the files you want to edit: `llmcode <file1> <file2> ...`
- Ask for changes:
  - Add new features or test cases.
  - Describe a bug.
  - Paste in an error message or GitHub issue URL.
  - Refactor code.
  - Update docs.
- Llmcode will edit your files to complete your request.
- Llmcode [automatically git commits](https://llmcode.khulnasoft.com/docs/git.html) changes with a sensible commit message.
- [Use llmcode inside your favorite editor or IDE](https://llmcode.khulnasoft.com/docs/usage/watch.html).
- Llmcode works with [most popular languages](https://llmcode.khulnasoft.com/docs/languages.html): python, javascript, typescript, php, html, css, and more...
- Llmcode can edit multiple files at once for complex requests.
- Llmcode uses a [map of your entire git repo](https://llmcode.khulnasoft.com/docs/repomap.html), which helps it work well in larger codebases.
- Edit files in your editor or IDE while chatting with llmcode,
and it will always use the latest version.
Pair program with AI.
- [Add images to the chat](https://llmcode.khulnasoft.com/docs/usage/images-urls.html) (GPT-4o, Claude 3.5 Sonnet, etc).
- [Add URLs to the chat](https://llmcode.khulnasoft.com/docs/usage/images-urls.html) and llmcode will read their content.
- [Code with your voice](https://llmcode.khulnasoft.com/docs/usage/voice.html).
- Llmcode works best with Claude 3.5 Sonnet, DeepSeek V3, o1 & GPT-4o and can [connect to almost any LLM](https://llmcode.khulnasoft.com/docs/llms.html).


## Top tier performance

[Llmcode has one of the top scores on SWE Bench](https://llmcode.khulnasoft.com/2024/06/02/main-swe-bench.html).
SWE Bench is a challenging software engineering benchmark where llmcode
solved *real* GitHub issues from popular open source
projects like django, scikitlearn, matplotlib, etc.

## More info

- [Documentation](https://llmcode.khulnasoft.com/)
- [Installation](https://llmcode.khulnasoft.com/docs/install.html)
- [Usage](https://llmcode.khulnasoft.com/docs/usage.html)
- [Tutorial videos](https://llmcode.khulnasoft.com/docs/usage/tutorials.html)
- [Connecting to LLMs](https://llmcode.khulnasoft.com/docs/llms.html)
- [Configuration](https://llmcode.khulnasoft.com/docs/config.html)
- [Troubleshooting](https://llmcode.khulnasoft.com/docs/troubleshooting.html)
- [LLM Leaderboards](https://llmcode.khulnasoft.com/docs/leaderboards/)
- [GitHub](https://github.com/KhulnaSoft/llmcode)
- [Discord](https://discord.gg/Tv2uQnR88V)
- [Blog](https://llmcode.khulnasoft.com/blog/)


## Kind words from users

- *The best free open source AI coding assistant.* -- [IndyDevDan](https://youtu.be/YALpX8oOn78)
- *The best AI coding assistant so far.* -- [Matthew Berman](https://www.youtube.com/watch?v=df8afeb1FY8)
- *Llmcode ... has easily quadrupled my coding productivity.* -- [SOLAR_FIELDS](https://news.ycombinator.com/item?id=36212100)
- *It's a cool workflow... Llmcode's ergonomics are perfect for me.* -- [qup](https://news.ycombinator.com/item?id=38185326)
- *It's really like having your senior developer live right in your Git repo - truly amazing!* -- [rappster](https://github.com/KhulnaSoft/llmcode/issues/124)
- *What an amazing tool. It's incredible.* -- [valyagolev](https://github.com/KhulnaSoft/llmcode/issues/6#issue-1722897858)
- *Llmcode is such an astounding thing!* -- [cgrothaus](https://github.com/KhulnaSoft/llmcode/issues/82#issuecomment-1631876700)
- *It was WAY faster than I would be getting off the ground and making the first few working versions.* -- [Daniel Feldman](https://twitter.com/d_feldman/status/1662295077387923456)
- *THANK YOU for Llmcode! It really feels like a glimpse into the future of coding.* -- [derwiki](https://news.ycombinator.com/item?id=38205643)
- *It's just amazing.  It is freeing me to do things I felt were out my comfort zone before.* -- [Dougie](https://discord.com/channels/1131200896827654144/1174002618058678323/1174084556257775656)
- *This project is stellar.* -- [funkytaco](https://github.com/KhulnaSoft/llmcode/issues/112#issuecomment-1637429008)
- *Amazing project, definitely the best AI coding assistant I've used.* -- [joshuavial](https://github.com/KhulnaSoft/llmcode/issues/84)
- *I absolutely love using Llmcode ... It makes software development feel so much lighter as an experience.* -- [principalideal0](https://discord.com/channels/1131200896827654144/1133421607499595858/1229689636012691468)
- *I have been recovering from multiple shoulder surgeries ... and have used llmcode extensively. It has allowed me to continue productivity.* -- [codeninja](https://www.reddit.com/r/OpenAI/s/nmNwkHy1zG)
- *I am an llmcode addict. I'm getting so much more work done, but in less time.* -- [dandandan](https://discord.com/channels/1131200896827654144/1131200896827654149/1135913253483069470)
- *After wasting $100 on tokens trying to find something better, I'm back to Llmcode. It blows everything else out of the water hands down, there's no competition whatsoever.* -- [SystemSculpt](https://discord.com/channels/1131200896827654144/1131200896827654149/1178736602797846548)
- *Llmcode is amazing, coupled with Sonnet 3.5 it’s quite mind blowing.* -- [Josh Dingus](https://discord.com/channels/1131200896827654144/1133060684540813372/1262374225298198548)
- *Hands down, this is the best AI coding assistant tool so far.* -- [IndyDevDan](https://www.youtube.com/watch?v=MPYFPvxfGZs)
- *[Llmcode] changed my daily coding workflows. It's mind-blowing how a single Python application can change your life.* -- [maledorak](https://discord.com/channels/1131200896827654144/1131200896827654149/1258453375620747264)
- *Best agent for actual dev work in existing codebases.* -- [Nick Dobos](https://twitter.com/NickADobos/status/1690408967963652097?s=20)
