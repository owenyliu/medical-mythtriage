"""A real LM for the agent, via OpenRouter — same plug shape as FakeLM.

The whole agent talks to its model through ONE callable:

    lm(messages) -> str        # messages = [{"role": ..., "content": ...}]

That is HW1's convention (cs329z_hw1/llm.py), so this adapter is a drop-in
replacement for `fake_lm.FakeLM`:

    from mythtriage.openrouter_lm import make_openrouter_lm
    result = verify_claim(claim, make_openrouter_lm(), corpus)

Key handling mirrors HW1 too: the key is read from the OPENROUTER_API_KEY
environment variable, which may come from a local `.env` file. The key
value is NEVER printed, logged, returned by any helper, or written anywhere
by this module — diagnostics use `shape_check()`, which reports only the
key's length and whether it starts with the expected "sk-or-" prefix.

.env search order (first existing file wins for each variable):
  1. the path in env var MYTHTRIAGE_ENV_FILE  (point this at your HW1
     .env and the key never has to be copied anywhere new)
  2. ./.env in the current working directory
  3. a .env next to the agent/ folder (this file's parent's parent)

Stdlib only: urllib for HTTP, so there is nothing to pip install.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_VAR = "OPENROUTER_API_KEY"
ENV_FILE_VAR = "MYTHTRIAGE_ENV_FILE"
EXPECTED_PREFIX = "sk-or-"

# Any OpenRouter model id works — this is just a cheap, widely available
# default. Override with the demo's --model flag or the argument below.
DEFAULT_MODEL = "openai/gpt-4o-mini"


# --------------------------------------------------------------------------
# .env loading — values go into os.environ only; they are never returned.
# --------------------------------------------------------------------------

def env_search_paths():
    """Where get_api_key() looks for a .env file, in order (paths only)."""
    candidates = []
    named = os.environ.get(ENV_FILE_VAR) or os.environ.get("RUMOR_AGENT_ENV_FILE")
    if named:
        candidates.append(Path(named).expanduser())
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parent.parent / ".env")
    paths, seen = [], set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:  # running from agent/ makes 2 and 3 the same
            seen.add(resolved)
            paths.append(path)
    return paths


def load_env_file(path):
    """Parse one .env file into os.environ. Returns how many NEW vars were set.

    Format: one KEY=VALUE per line; blank lines and `#` comments ignored;
    an optional leading `export ` and optional quotes around the value are
    accepted. A variable already present in the environment is NOT
    overwritten — the real environment always beats a file.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        return 0
    set_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            set_count += 1
    return set_count


def _load_env_files():
    for path in env_search_paths():
        load_env_file(path)


# --------------------------------------------------------------------------
# Key access — the value itself only ever travels into the request header.
# --------------------------------------------------------------------------

def get_api_key():
    """Return the OpenRouter key from the environment (after .env loading),
    or None if it is not set anywhere we looked."""
    _load_env_files()
    return os.environ.get(KEY_VAR) or None


def shape_check(key):
    """(length, starts_with_expected_prefix) — safe to print; never the key."""
    if not key:
        return (0, False)
    return (len(key), key.startswith(EXPECTED_PREFIX))


def require_api_key():
    """get_api_key(), or raise a RuntimeError that says where we looked."""
    key = get_api_key()
    if key:
        return key
    looked = "; ".join(str(p) for p in env_search_paths())
    raise RuntimeError(
        f"{KEY_VAR} is not set. It was looked for in the environment and in "
        f".env files at: {looked}. Point {ENV_FILE_VAR} at your existing "
        f".env (e.g. your HW1 repo's), or create a .env with a line "
        f"{KEY_VAR}=<your key>. The key must never be pasted into chat or "
        f"hardcoded in code."
    )


# --------------------------------------------------------------------------
# The LM callable.
# --------------------------------------------------------------------------

def _urllib_transport(url, headers, body):
    """Default transport: a real POST via urllib. `transport` in
    make_openrouter_lm has this exact signature so tests can inject a fake
    and run fully offline."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Status code ONLY in the message — an error page could contain
        # anything, and the request carried the key in its headers.
        raise RuntimeError(
            f"OpenRouter request failed with HTTP status {exc.code}. "
            f"Check the model name and that the key is valid."
        ) from None
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"OpenRouter request failed: could not reach the server "
            f"({exc.reason})."
        ) from None


def make_openrouter_lm(model=DEFAULT_MODEL, transport=None):
    """Build an `lm(messages) -> str` backed by OpenRouter.

    model:     any OpenRouter model id, e.g. "openai/gpt-4o-mini".
    transport: optional callable (url, headers, body_dict) -> response_dict,
               injected by tests; defaults to a real urllib POST.

    The key is resolved on EVERY call, so a key loaded after this factory
    runs is still picked up. A missing key raises RuntimeError — there is
    deliberately NO silent fallback to FakeLM: a live run must never
    quietly become a fake one. Temperature is fixed at 0 so evaluation
    runs are deterministic.
    """
    send = transport if transport is not None else _urllib_transport

    def lm(messages):
        key = require_api_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "CS329Z Medical MythTriage",
        }
        body = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]}
                         for m in messages],
            "temperature": 0,
        }
        try:
            response = send(OPENROUTER_URL, headers, body)
        except urllib.error.HTTPError as exc:
            # An injected transport may let the raw error through; the same
            # rule applies: status code only, never headers or body.
            raise RuntimeError(
                f"OpenRouter request failed with HTTP status {exc.code}."
            ) from None
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "OpenRouter returned an unexpected response shape: "
                "no choices[0].message.content."
            ) from None

    lm.model = model
    return lm
