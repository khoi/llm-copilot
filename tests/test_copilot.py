import json
import pathlib
import pytest
from pytest_httpx import IteratorStream
import llm
import time

TEST_MODELS = {
    "data": [
        {
            "capabilities": {
                "family": "gpt-4o",
                "limits": {
                    "max_context_window_tokens": 128000,
                    "max_output_tokens": 4096,
                    "max_prompt_tokens": 64000,
                },
                "object": "model_capabilities",
                "supports": {"parallel_tool_calls": True, "tool_calls": True},
                "tokenizer": "o200k_base",
                "type": "chat",
            },
            "id": "gpt-4o",
            "model_picker_enabled": True,
            "name": "GPT 4o",
            "object": "model",
            "preview": False,
            "vendor": "Azure OpenAI",
            "version": "gpt-4o-2024-05-13",
        },
        {
            "capabilities": {
                "family": "gpt-4o-mini",
                "limits": {
                    "max_context_window_tokens": 128000,
                    "max_output_tokens": 4096,
                    "max_prompt_tokens": 12288,
                },
                "object": "model_capabilities",
                "supports": {"parallel_tool_calls": True, "tool_calls": True},
                "tokenizer": "o200k_base",
                "type": "chat",
            },
            "id": "gpt-4o-mini",
            "model_picker_enabled": False,
            "name": "GPT 4o Mini",
            "object": "model",
            "preview": False,
            "vendor": "Azure OpenAI",
            "version": "gpt-4o-mini-2024-07-18",
        },
        {
            "capabilities": {"type": "embeddings"},
            "id": "text-embedding-ada-002",
            "name": "Embedding Model",
            "object": "model",
        },
    ],
    "object": "list",
}


@pytest.fixture(scope="session")
def llm_user_path(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("llm")
    return str(tmpdir)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, llm_user_path):
    monkeypatch.setenv("LLM_GITHUB_COPILOT_KEY", "test_key")
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)


@pytest.mark.httpx_mock(assert_all_requests_were_expected=False)
def test_caches_models(monkeypatch, tmpdir, httpx_mock):
    httpx_mock.add_response(
        url="https://api.business.githubcopilot.com/models",
        method="GET",
        json=TEST_MODELS,
    )
    httpx_mock.add_response(
        url="https://api.github.com/copilot_internal/v2/token",
        method="GET",
        json={"token": "test_token", "expires_at": time.time() + 3600},
    )

    llm_user_path = str(tmpdir / "llm")
    monkeypatch.setenv("LLM_USER_PATH", llm_user_path)

    models_path = pathlib.Path(llm_user_path) / "llm-copilot-models.json"
    assert not models_path.exists()

    models_with_aliases = llm.get_models_with_aliases()
    assert models_path.exists()

    request = [r for r in httpx_mock.get_requests() if str(r.url).endswith("/models")][
        0
    ]
    assert request.url == "https://api.business.githubcopilot.com/models"


@pytest.fixture
def mocked_stream(httpx_mock):
    httpx_mock.add_response(
        url="https://api.github.com/copilot_internal/v2/token",
        method="GET",
        json={"token": "test_token", "expires_at": time.time() + 3600},
        is_optional=True,
    )
    # Mock streaming endpoint
    httpx_mock.add_response(
        url="https://api.business.githubcopilot.com/chat/completions",
        method="POST",
        stream=IteratorStream(
            [
                b'data: {"choices":[],"created":0,"id":"","prompt_filter_results":[{"content_filter_results":{"error":{"code":"","message":""},"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"prompt_index":0}]}\n\n',
                b'data: {"choices":[{"index":0,"content_filter_offsets":{"check_offset":9528,"start_offset":9528,"end_offset":9535},"content_filter_results":{"error":{"code":"","message":""},"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"delta":{"content":"","role":"assistant"}}],"created":1732435067,"id":"chatcmpl-AX1n1QIOwXCocpUmBCZOI8r6P4usG","model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_04751d0b65"}\n\n',
                b'data: {"choices":[{"index":0,"content_filter_offsets":{"check_offset":9528,"start_offset":9528,"end_offset":9535},"content_filter_results":{"error":{"code":"","message":""},"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"delta":{"content":"unknown"}}],"created":1732435067,"id":"chatcmpl-AX1n1QIOwXCocpUmBCZOI8r6P4usG","model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_04751d0b65"}\n\n',
                b'data: {"choices":[{"finish_reason":"stop","index":0,"content_filter_offsets":{"check_offset":9528,"start_offset":9528,"end_offset":9535},"content_filter_results":{"error":{"code":"","message":""},"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}},"delta":{"content":null}}],"created":1732435067,"id":"chatcmpl-AX1n1QIOwXCocpUmBCZOI8r6P4usG","usage":{"completion_tokens":1,"prompt_tokens":2301,"total_tokens":2302},"model":"gpt-4o-mini-2024-07-18","system_fingerprint":"fp_04751d0b65"}\n\n',
                b"data: [DONE]\n",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )
    return httpx_mock


@pytest.fixture
def mocked_no_stream(httpx_mock):
    httpx_mock.add_response(
        url="https://api.github.com/copilot_internal/v2/token",
        method="GET",
        json={"token": "test_token", "expires_at": time.time() + 3600},
        is_optional=True,
    )
    # Mock non-streaming endpoint
    httpx_mock.add_response(
        url="https://api.business.githubcopilot.com/chat/completions",
        method="POST",
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm GitHub Copilot, an AI coding assistant.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 16, "total_tokens": 79, "completion_tokens": 63},
        },
    )
    return httpx_mock


@pytest.mark.httpx_mock(assert_all_requests_were_expected=False)
def test_stream(mocked_stream):
    model = llm.get_model("copilot-gpt-4o")
    response = model.prompt("How are you?")
    chunks = list(response)
    assert chunks == ["unknown"]

    request = [
        r for r in mocked_stream.get_requests() if "/chat/completions" in str(r.url)
    ][0]
    assert json.loads(request.content) == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.1,
        "top_p": 1,
        "max_tokens": 4096,
        "stream": True,
        "n": 1,
    }


@pytest.mark.httpx_mock(assert_all_requests_were_expected=False)
def test_no_stream(mocked_no_stream):
    model = llm.get_model("copilot-gpt-4o")
    response = model.prompt("How are you?", stream=False)
    assert response.text() == "I'm GitHub Copilot, an AI coding assistant."

    request = [
        r for r in mocked_no_stream.get_requests() if "/chat/completions" in str(r.url)
    ][0]
    assert json.loads(request.content) == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.1,
        "top_p": 1,
        "max_tokens": 4096,
        "stream": False,
        "n": 1,
    }


@pytest.mark.httpx_mock(assert_all_requests_were_expected=False)
def test_stream_with_options(mocked_stream):
    model = llm.get_model("copilot-gpt-4o")
    model.prompt(
        "How are you?", temperature=0.5, top_p=0.8, max_tokens=10, stop=[".", "\n"], n=2
    ).text()

    request = [
        r for r in mocked_stream.get_requests() if "/chat/completions" in str(r.url)
    ][0]
    assert json.loads(request.content) == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.5,
        "top_p": 0.8,
        "max_tokens": 10,
        "stop": [".", "\n"],
        "n": 2,
        "stream": True,
    }
