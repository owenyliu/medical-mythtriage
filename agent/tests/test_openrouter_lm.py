"""Offline tests for the OpenRouter LM adapter.

No network, no real key: every test uses an injected fake transport and
the obviously-dummy key "sk-or-testdummy". Run from the agent/ folder:

    python -m pytest tests/ -q
"""

import io
import os
import urllib.error

import pytest

from mythtriage.openrouter_lm import (
    DEFAULT_MODEL,
    OPENROUTER_URL,
    get_api_key,
    load_env_file,
    make_openrouter_lm,
    shape_check,
)

DUMMY_KEY = "sk-or-testdummy"
MESSAGES = [{"role": "system", "content": "TASK: GLANCE"},
            {"role": "user", "content": "Claim: a test claim"}]
CANNED = {"choices": [{"message": {"content": "reply from the fake transport"}}]}


def recording_transport(calls, response=CANNED):
    """A transport stub: remembers (url, headers, body), returns `response`."""
    def transport(url, headers, body):
        calls.append((url, headers, body))
        return response
    return transport


# -- request shape + response parsing ---------------------------------------

def test_request_url_auth_body_and_parsing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)
    calls = []
    lm = make_openrouter_lm(transport=recording_transport(calls))

    reply = lm(MESSAGES)

    assert reply == "reply from the fake transport"
    assert len(calls) == 1
    url, headers, body = calls[0]
    assert url == OPENROUTER_URL
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {DUMMY_KEY}"
    assert body["model"] == DEFAULT_MODEL
    assert body["messages"] == MESSAGES
    assert body["temperature"] == 0  # deterministic for evals


def test_model_argument_is_used(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)
    calls = []
    lm = make_openrouter_lm(model="anthropic/claude-haiku-4.5",
                            transport=recording_transport(calls))
    lm(MESSAGES)
    assert calls[0][2]["model"] == "anthropic/claude-haiku-4.5"
    assert lm.model == "anthropic/claude-haiku-4.5"


def test_malformed_response_raises_clearly(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)
    lm = make_openrouter_lm(transport=recording_transport([], response={}))
    with pytest.raises(RuntimeError, match="unexpected response shape"):
        lm(MESSAGES)


# -- missing key --------------------------------------------------------------

def test_missing_key_raises_and_names_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Point the search at empty locations so no real .env can leak in.
    monkeypatch.setenv("MYTHTRIAGE_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.chdir(tmp_path)

    def forbidden_transport(url, headers, body):  # must never be reached
        raise AssertionError("transport called without a key")

    lm = make_openrouter_lm(transport=forbidden_transport)
    with pytest.raises(RuntimeError) as excinfo:
        lm(MESSAGES)
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


# -- .env parsing ---------------------------------------------------------------

def test_load_env_file_parses_and_never_overrides(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment line\n"
        "\n"
        f"OPENROUTER_API_KEY={DUMMY_KEY}\n"
        'QUOTED_KEY="quoted value"\n'
        "export EXPORTED_KEY=exported-value\n"
        "PLAIN_KEY=from-file\n"
        "a line without an equals sign\n",
        encoding="utf-8",
    )
    for var in ("OPENROUTER_API_KEY", "QUOTED_KEY", "EXPORTED_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PLAIN_KEY", "preexisting")

    set_count = load_env_file(env_file)

    assert set_count == 3  # PLAIN_KEY already existed -> skipped, not counted
    assert os.environ["OPENROUTER_API_KEY"] == DUMMY_KEY
    assert os.environ["QUOTED_KEY"] == "quoted value"
    assert os.environ["EXPORTED_KEY"] == "exported-value"
    assert os.environ["PLAIN_KEY"] == "preexisting"  # environment wins


def test_get_api_key_finds_key_via_env_file_variable(monkeypatch, tmp_path):
    env_file = tmp_path / "hw1.env"
    env_file.write_text(f"OPENROUTER_API_KEY={DUMMY_KEY}\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MYTHTRIAGE_ENV_FILE", str(env_file))
    assert get_api_key() == DUMMY_KEY


def test_load_env_file_missing_path_is_a_noop(tmp_path):
    assert load_env_file(tmp_path / "does-not-exist.env") == 0


# -- diagnostics never leak the key --------------------------------------------

def test_shape_check_reports_shape_only():
    assert shape_check(DUMMY_KEY) == (len(DUMMY_KEY), True)
    assert shape_check("not-an-openrouter-key") == (21, False)
    assert shape_check(None) == (0, False)
    assert shape_check("") == (0, False)


def test_http_error_from_transport_has_status_but_not_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)

    def leaky_transport(url, headers, body):
        # Simulate a server error page that (badly) echoes the key back.
        raise urllib.error.HTTPError(
            url, 401, "Unauthorized", {},
            io.BytesIO(f"bad key: {DUMMY_KEY}".encode()))

    lm = make_openrouter_lm(transport=leaky_transport)
    with pytest.raises(RuntimeError) as excinfo:
        lm(MESSAGES)
    message = str(excinfo.value)
    assert "401" in message
    assert DUMMY_KEY not in message


def test_default_transport_http_error_has_status_but_not_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", {},
            io.BytesIO(f"slow down, {DUMMY_KEY}".encode()))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lm = make_openrouter_lm()  # no injected transport -> urllib path
    with pytest.raises(RuntimeError) as excinfo:
        lm(MESSAGES)
    message = str(excinfo.value)
    assert "429" in message
    assert DUMMY_KEY not in message
