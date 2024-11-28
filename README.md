## llm-copilot

A plugin for [LLM](https://llm.datasette.io) that allows you to use ur existing Github Copilot Enterprise.

## Installation

I'll have to publish this to PyPi first, for now.

- Install [LLM](https://llm.datasette.io)
- Clone
- `llm install -e .`

## Caveats

The plugin requires you to have a Github Copilot API Token, this is different from your Github Token.

The easiest way to fetch it is to setup <https://github.com/github/CopilotForXcode>

Once that's working the plugin will piggy back on the token setup for Xcode.

I will add a more direct way to fetch the token in the future.

