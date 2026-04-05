import pytest

from agent.smart_model_routing import resolve_turn_route
from hermes_constants import parse_reasoning_effort



def _primary():
    return {
        "model": "gpt-5.4",
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
        "reasoning_config": parse_reasoning_effort("xhigh"),
    }



def _cfg():
    return {
        "enabled": True,
        "max_simple_chars": 600,
        "max_simple_words": 96,
        "cheap_model": {
            "provider": "openai-codex",
            "model": "gpt-5.4-mini",
            "reasoning_effort": "xhigh",
        },
    }



def _mock_runtime_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested, explicit_api_key=None, explicit_base_url=None: {
            "provider": requested,
            "api_key": explicit_api_key or "test-key",
            "base_url": explicit_base_url or "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
        },
    )



def test_conversational_turn_routes_cheap(monkeypatch):
    _mock_runtime_provider(monkeypatch)
    result = resolve_turn_route("are you using the cheaper model for normal chat?", _cfg(), _primary())
    assert result["model"] == "gpt-5.4-mini"
    assert result["routing_reason"] == "cheap_default_non_complex_turn"
    assert result["reasoning_config"] == {"enabled": True, "effort": "xhigh"}



def test_password_turn_routes_cheap(monkeypatch):
    _mock_runtime_provider(monkeypatch)
    result = resolve_turn_route("what's the password?", _cfg(), _primary())
    assert result["model"] == "gpt-5.4-mini"
    assert result["routing_reason"] == "cheap_default_non_complex_turn"



def test_env_or_debug_turn_stays_primary():
    result = resolve_turn_route("can you check if these env vars are all cool?", _cfg(), _primary())
    assert result["model"] == "gpt-5.4"
    assert result["routing_reason"] == "primary_default"
    assert result["reasoning_config"] == {"enabled": True, "effort": "xhigh"}



def test_explicit_planning_request_stays_primary():
    result = resolve_turn_route("can you plan the rollout for this feature?", _cfg(), _primary())
    assert result["model"] == "gpt-5.4"
    assert result["routing_reason"] == "primary_default"



def test_long_multiline_turn_stays_primary():
    message = "one\ntwo\nthree\nfour"
    result = resolve_turn_route(message, _cfg(), _primary())
    assert result["model"] == "gpt-5.4"
    assert result["routing_reason"] == "primary_default"
