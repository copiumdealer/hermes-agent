from unittest.mock import MagicMock

from agent.smart_model_routing import choose_cheap_model_route, resolve_turn_route


_BASE_CONFIG = {
    "enabled": True,
    "max_simple_chars": 600,
    "max_simple_words": 96,
    "cheap_model": {
        "provider": "openai-codex",
        "model": "gpt-5.4-mini",
        "reasoning_effort": "xhigh",
    },
}


def _primary():
    return {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
        "api_key": "test-key",
        "credential_pool": object(),
        "reasoning_config": {"enabled": True, "effort": "xhigh"},
    }


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None



def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openai-codex"
    assert result["model"] == "gpt-5.4-mini"
    assert result["routing_reason"] == "cheap_default_non_complex_turn"



def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 40
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None



def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None



def test_skips_tool_heavy_prompt_keywords():
    prompt = "implement a patch for this docker error"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None



def test_same_provider_route_reuses_primary_runtime_and_pool(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        MagicMock(side_effect=AssertionError("should not resolve runtime for same-provider cheap route")),
    )
    primary = _primary()

    result = resolve_turn_route("got it, thanks", _BASE_CONFIG, primary)

    assert result["model"] == "gpt-5.4-mini"
    assert result["runtime"]["provider"] == "openai-codex"
    assert result["runtime"]["base_url"] == primary["base_url"]
    assert result["runtime"]["credential_pool"] is primary["credential_pool"]
    assert result["reasoning_config"] == {"enabled": True, "effort": "xhigh"}



def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "hi",
        {
            "enabled": True,
            "cheap_model": {
                "provider": "openrouter",
                "model": "openai/gpt-4.1-mini",
            },
            "max_simple_chars": 600,
            "max_simple_words": 96,
        },
        _primary(),
    )
    assert result["model"] == "gpt-5.4"
    assert result["runtime"]["provider"] == "openai-codex"
    assert result["label"] is None
    assert result["routing_reason"] == "primary_fallback"



def test_legacy_config_shape_still_routes():
    result = choose_cheap_model_route(
        "hello there",
        {
            "enabled": True,
            "cheap_model": "gpt-5.4-mini",
            "cheap_provider": "openai-codex",
        },
    )
    assert result is not None
    assert result["provider"] == "openai-codex"
    assert result["model"] == "gpt-5.4-mini"
