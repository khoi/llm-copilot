import llm
import os
import json
from typing import Optional, List
from pydantic import Field
import httpx
import time
from httpx_sse import connect_sse, aconnect_sse

DEFAULT_ALIASES = {
    "copilot-gpt-3.5-turbo": "gpt-3.5-turbo",
    "copilot-gpt-4": "gpt-4",
    "copilot-gpt-4o": "gpt-4",
    "copilot-gpt-4o-mini": "gpt-4",
}

DEFAULT_MODELS = list(DEFAULT_ALIASES.keys())

BASE_URL = "https://api.business.githubcopilot.com"
TOKEN_PATH = os.path.expanduser("~/.config/github-copilot/llm-copilot-token.json")


@llm.hookimpl
def register_models(register):
    for model_id in DEFAULT_MODELS:
        register(Copilot(model_id))


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="copilot")
    def copilot():
        "Commands for Github Copilot models"


class _Shared:
    needs_key = "github-copilot"
    can_stream = True

    class Options(llm.Options):
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

    def infer_key(self):
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
        headers = {
            "Authorization": f"Bearer {oauth_token}",
            "Accept": "application/json",
        }

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
            "model": DEFAULT_ALIASES[self.model_id],
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


class Copilot(_Shared, llm.Model):
    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation=None,
    ):
        oauth_token = self.infer_key() or self.get_key()
        auth_token = self.authorize_token(oauth_token)
        messages = self.build_messages(prompt, conversation)
        body = self.build_body(prompt, messages, stream)

        if stream:
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    f"{BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {auth_token['token']}",
                        "Copilot-Integration-Id": "vscode-chat",
                        "editor-plugin-version": "copilot-chat/0.22.4",
                        "editor-version": "vscode/1.95.3",
                        "user-agent": "GitHubCopilotChat/0.22.4",
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
                    f"{BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
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
