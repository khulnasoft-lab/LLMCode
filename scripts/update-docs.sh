#!/bin/bash

# exit when any command fails
set -e

if [ -z "$1" ]; then
  ARG=-r
else
  ARG=$1
fi

if [ "$ARG" != "--check" ]; then
  tail -1000 ~/.llmcode/analytics.jsonl > llmcode/website/assets/sample-analytics.jsonl
  cog -r llmcode/website/docs/faq.md
fi

# README.md before index.md, because index.md uses cog to include README.md
cog $ARG \
    README.md \
    llmcode/website/index.html \
    llmcode/website/HISTORY.md \
    llmcode/website/docs/usage/commands.md \
    llmcode/website/docs/languages.md \
    llmcode/website/docs/config/dotenv.md \
    llmcode/website/docs/config/options.md \
    llmcode/website/docs/config/llmcode_conf.md \
    llmcode/website/docs/config/adv-model-settings.md \
    llmcode/website/docs/config/model-aliases.md \
    llmcode/website/docs/leaderboards/index.md \
    llmcode/website/docs/leaderboards/edit.md \
    llmcode/website/docs/leaderboards/refactor.md \
    llmcode/website/docs/llms/other.md \
    llmcode/website/docs/more/infinite-output.md \
    llmcode/website/docs/legal/privacy.md
