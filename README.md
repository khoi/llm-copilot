## llm-copilot

A plugin for [LLM](https://llm.datasette.io) that allows you to use ur existing Github Copilot Enterprise.

## Installation

My account at PyPi is still waiting for approval. Since I couldn't publish it yet. Here is a development way of installing the plugin.

- Install [LLM](https://llm.datasette.io)
- Clone this repo to a folder
- `llm install -e .`

## Usage

The plugin will try to piggy-back on existing Github Copilot extensions (Xcode Copilot, Copilot.nvim) tokens stored in `~/.config/github-copilot/`

Since you're most likely coming from Goodnotes 👋

- Install <https://github.com/github/CopilotForXcode> and set it up
- This plugin will use the token there.

To verify it's working run

```bash
llm models # See if the list of models contain copilot-* models
llm --model "copilot-gpt-4o" "Hello world"
```

## TODOs

- [ ] Publish to PyPi
- [ ] Add a way to fetch token from Github.com directly
