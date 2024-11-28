import click
import llm
import os
import json
from pathlib import Path
from typing import Optional, List
from pydantic import Field
import httpx
import time
from httpx_sse import connect_sse

DEFAULT_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
}

TOKEN_PATH = llm.user_dir() / "llm-copilot-token.json"
MODELS_PATH = llm.user_dir() / "llm-copilot-models.json"

COPILOT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Copilot-Integration-Id": "vscode-chat",
    "editor-plugin-version": "copilot-chat/0.22.4",
    "editor-version": "vscode/1.95.3",
    "user-agent": "GitHubCopilotChat/0.22.4",
}


def filter_chat_models(models_data):
    """Extract chat model IDs from models data response."""
    return [
        model["id"]
        for model in models_data["data"]
        if model.get("capabilities", {}).get("type") == "chat"
    ]


def infer_key():
    """Infer GitHub Copilot token from config files."""
    config_path = os.path.expanduser("~/.config")
    file_paths = [
        os.path.join(config_path, "github-copilot/hosts.json"),
        os.path.join(config_path, "github-copilot/apps.json"),
    ]

    for file_path in file_paths:
        if os.path.isfile(file_path):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                for key, value in data.items():
                    if "github.com" in key:
                        return value.get("oauth_token")
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
    return None


def refresh_models():
    """Refresh the list of available Copilot models"""
    models_file = Path(MODELS_PATH)
    oauth_token = infer_key() or llm.get_key("", "github-copilot")
    if not oauth_token:
        raise click.ClickException(
            "You must set the 'github-copilot' key or have valid GitHub Copilot credentials."
        )

    # First get auth token
    headers = {**COPILOT_HEADERS, "Authorization": f"Bearer {oauth_token}"}
    response = httpx.get(
        "https://api.github.com/copilot_internal/v2/token",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    auth_token = response.json()["token"]

    # Then get models
    response = httpx.get(
        "https://api.business.githubcopilot.com/models",
        headers={**COPILOT_HEADERS, "Authorization": f"Bearer {auth_token}"},
    )
    response.raise_for_status()
    models_data = response.json()

    # Filter to just chat models and store their IDs
    chat_models = filter_chat_models(models_data)

    # Store the full response for future reference
    models_file.write_text(json.dumps(models_data, indent=2))

    return {"models": chat_models}


def get_model_details():
    """Get cached model details or use defaults"""
    models = {"models": DEFAULT_MODELS}
    models_file = Path(MODELS_PATH)
    if models_file.exists():
        try:
            models_data = json.loads(models_file.read_text())
            # Filter to just chat models and store their IDs
            chat_models = filter_chat_models(models_data)
            models = {"models": chat_models}
        except (json.JSONDecodeError, KeyError):
            pass
    elif infer_key() or llm.get_key("", "github-copilot"):
        try:
            models = refresh_models()
        except Exception:
            pass

    return models.get("models", DEFAULT_MODELS)


@llm.hookimpl
def register_models(register):
    for model_id in get_model_details():
        register(Copilot(f"copilot-{model_id}"))


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="copilot")
    def copilot():
        "Commands for Github Copilot models"

    @copilot.command()
    def refresh():
        "Refresh the list of available Copilot models"
        before = set(get_model_details())
        refresh_models()
        after = set(get_model_details())
        added = after - before
        removed = before - after
        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("New list of models:", err=True)
            for model_id in get_model_details():
                click.echo(model_id, err=True)
        else:
            click.echo("No changes", err=True)


class Copilot(llm.Model):
    needs_key = "github-copilot"
    can_stream = True

    class Options(llm.Model.Options):
        max_tokens: Optional[int] = Field(
            default=4_096,
        )

        temperature: Optional[float] = Field(
            default=0.1,
        )

        top_p: Optional[float] = Field(
            default=1,
        )

        stop: Optional[List[str]] = Field(
            default=None,
        )

        n: Optional[int] = Field(
            default=1,
        )

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._api_model_id = model_id.replace("copilot-", "", 1)

    def ensure_config_dir(self):
        """Ensure the config directory exists."""
        config_dir = os.path.dirname(TOKEN_PATH)
        os.makedirs(config_dir, exist_ok=True)

    def authorize_token(self, oauth_token):
        """Authorize GitHub Copilot token and cache it."""
        # Try to load cached token
        try:
            if os.path.exists(TOKEN_PATH):
                with open(TOKEN_PATH) as f:
                    cached_token = json.load(f)
                if cached_token.get("expires_at", 0) > time.time():
                    return cached_token
        except (json.JSONDecodeError, KeyError):
            pass

        # Fetch new token
        headers = {**COPILOT_HEADERS, "Authorization": f"Bearer {oauth_token}"}

        response = httpx.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers=headers,
            timeout=30,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to authorize token: {response.text}")

        token_data = response.json()

        # Cache the token
        self.ensure_config_dir()
        with open(TOKEN_PATH, "w") as f:
            json.dump(token_data, f)

        return token_data

    def build_messages(self, prompt, conversation):
        messages = []
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            if prompt.attachments:
                messages[-1]["images"] = [
                    attachment.base64_content() for attachment in prompt.attachments
                ]
            return messages

        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system},
                )
                current_system = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            if prev_response.attachments:
                messages[-1]["images"] = [
                    attachment.base64_content()
                    for attachment in prev_response.attachments
                ]

            messages.append({"role": "assistant", "content": prev_response.text()})
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def build_body(self, prompt, messages, stream=True):
        body = {
            "model": self._api_model_id,
            "messages": messages,
            "max_tokens": prompt.options.max_tokens,
            "stream": stream,
        }

        if prompt.options.top_p:
            body["top_p"] = prompt.options.top_p

        if prompt.options.temperature:
            body["temperature"] = prompt.options.temperature

        if prompt.options.stop:
            body["stop"] = prompt.options.stop

        if prompt.options.n:
            body["n"] = prompt.options.n

        return body

    def set_usage(self, response, usage):
        response.set_usage(
            input=usage["prompt_tokens"],
            output=usage["completion_tokens"],
        )

    def __str__(self) -> str:
        return f"Copilot: {self.model_id}"

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation=None,
    ):
        oauth_token = infer_key() or self.get_key()
        auth_token = self.authorize_token(oauth_token)
        messages = self.build_messages(prompt, conversation)
        body = self.build_body(prompt, messages, stream)

        if stream:
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    "https://api.business.githubcopilot.com/chat/completions",
                    headers={
                        **COPILOT_HEADERS,
                        "Authorization": f"Bearer {auth_token['token']}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    if event_source.response.status_code != 200:
                        error_body = event_source.response.read().decode()
                        raise llm.ModelError(
                            f"HTTP Status {event_source.response.status_code}: {error_body}"
                        )

                    usage = None
                    event_source.response.raise_for_status()
                    # this is a hack to make httpx_see works. see https://github.com/florimondmanca/httpx-sse/pull/12
                    event_source.response.headers["content-type"] = "text/event-stream"
                    for sse in event_source.iter_sse():
                        if sse.data != "[DONE]":
                            try:
                                event = sse.json()
                                if "usage" in event:
                                    usage = event["usage"]
                                if event.get("choices") and event["choices"][0].get(
                                    "delta", {}
                                ).get("content"):
                                    yield event["choices"][0]["delta"]["content"]
                            except KeyError:
                                pass
                    if usage:
                        self.set_usage(response, usage)
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    "https://api.business.githubcopilot.com/chat/completions",
                    headers={
                        **COPILOT_HEADERS,
                        "Authorization": f"Bearer {auth_token['token']}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                yield api_response.json()["choices"][0]["message"]["content"]
                details = api_response.json()
                usage = details.pop("usage", None)
                response.response_json = details
                if usage:
                    self.set_usage(response, usage)
