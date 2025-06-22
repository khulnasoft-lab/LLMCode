#!/bin/bash

# exit when any command fails
set -e

if [ -z "$1" ]; then
  ARG=-r
else
  ARG=$1
fi

if [ "$ARG" != "--check" ]; then
  tail -1000 ~/.llmcode/analytics.jsonl > docs/site/assets/sample-analytics.jsonl
  cog -r docs/site/docs/faq.md
fi

# README.md before index.md, because index.md uses cog to include README.md
cog $ARG \
    README.md \
    docs/site/index.md \
    docs/site/HISTORY.md \
    docs/site/docs/usage/commands.md \
    docs/site/docs/languages.md \
    docs/site/docs/config/dotenv.md \
    docs/site/docs/config/options.md \
    docs/site/docs/config/llmcode_conf.md \
    docs/site/docs/config/adv-model-settings.md \
    docs/site/docs/config/model-aliases.md \
    docs/site/docs/leaderboards/index.md \
    docs/site/docs/leaderboards/edit.md \
    docs/site/docs/leaderboards/refactor.md \
    docs/site/docs/llms/other.md \
    docs/site/docs/more/infinite-output.md \
    docs/site/docs/legal/privacy.md
