"""Helpers for optional cheap-vs-strong model routing."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from hermes_constants import parse_reasoning_effort
from utils import is_truthy_value

_TASK_ARTIFACT_KEYWORDS = {
    "api",
    "apis",
    "auth",
    "bash",
    "branch",
    "bug",
    "build",
    "ci",
    "cd",
    "class",
    "classes",
    "code",
    "command",
    "commands",
    "commit",
    "commits",
    "config",
    "configs",
    "database",
    "databases",
    "debug",
    "deploy",
    "deployment",
    "dependency",
    "dependencies",
    "diff",
    "docker",
    "endpoint",
    "endpoints",
    "env",
    "environment",
    "error",
    "errors",
    "exception",
    "exceptions",
    "file",
    "files",
    "function",
    "functions",
    "git",
    "github",
    "infra",
    "json",
    "kubernetes",
    "library",
    "libraries",
    "log",
    "logs",
    "migration",
    "module",
    "modules",
    "oauth",
    "package",
    "packages",
    "patch",
    "postgres",
    "pr",
    "prompt",
    "prompts",
    "pytest",
    "python",
    "railway",
    "repo",
    "repos",
    "repository",
    "schema",
    "schemas",
    "script",
    "scripts",
    "sdk",
    "server",
    "service",
    "shell",
    "sql",
    "stacktrace",
    "terminal",
    "test",
    "tests",
    "token",
    "tokens",
    "tool",
    "tools",
    "traceback",
    "typescript",
    "workflow",
    "workflows",
    "yaml",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_DIRECT_STRONG_REQUEST_RE = re.compile(
    r"^\s*(debug|fix|implement|refactor|patch|trace|investigate|triage|analy[sz]e|compare|benchmark|optimi[sz]e|review|research|plan|design|architect|build|deploy)\b",
    re.IGNORECASE,
)
_ASSISTED_STRONG_REQUEST_RE = re.compile(
    r"\b(can you|could you|would you|please|i need you to|need you to|help me|let's|lets)\b.{0,64}\b(debug|fix|implement|refactor|patch|trace|investigate|triage|analy[sz]e|compare|benchmark|optimi[sz]e|review|research|plan|design|architect|build|deploy)\b",
    re.IGNORECASE | re.DOTALL,
)
_FAILURE_TRIAGE_RE = re.compile(
    r"\b(why|how|what)\b.{0,64}\b(fail|failed|failing|broken|crash|crashing|timeout|timed out|error|bug)\b",
    re.IGNORECASE | re.DOTALL,
)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_words(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}\"'`").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}\"'`")
    }


def _looks_like_complex_task(text: str) -> bool:
    lowered = text.lower()
    words = _normalize_words(text)
    if words & _TASK_ARTIFACT_KEYWORDS:
        return True
    if _DIRECT_STRONG_REQUEST_RE.search(text):
        return True
    if _ASSISTED_STRONG_REQUEST_RE.search(text):
        return True
    if _FAILURE_TRIAGE_RE.search(text):
        return True
    if text.count("?") > 1:
        return True
    if lowered.count(" step ") or lowered.startswith("step "):
        return True
    return False


def _normalize_cheap_model_config(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cheap_model = cfg.get("cheap_model") or {}
    if isinstance(cheap_model, str):
        normalized: Dict[str, Any] = {"model": cheap_model}
        if cfg.get("cheap_provider"):
            normalized["provider"] = cfg.get("cheap_provider")
        if cfg.get("cheap_base_url"):
            normalized["base_url"] = cfg.get("cheap_base_url")
        if cfg.get("cheap_api_key_env"):
            normalized["api_key_env"] = cfg.get("cheap_api_key_env")
        if cfg.get("cheap_reasoning_effort"):
            normalized["reasoning_effort"] = cfg.get("cheap_reasoning_effort")
        cheap_model = normalized
    if not isinstance(cheap_model, dict):
        return None
    return cheap_model


def _route_reasoning_config(route: Dict[str, Any], primary: Dict[str, Any]) -> Any:
    effort = str(route.get("reasoning_effort") or route.get("reasoning") or "").strip()
    if effort:
        parsed = parse_reasoning_effort(effort)
        if parsed is not None:
            return parsed
    return primary.get("reasoning_config")


def _route_signature(model: Any, runtime: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        model,
        runtime.get("provider"),
        runtime.get("base_url"),
        runtime.get("api_mode"),
        runtime.get("command"),
        tuple(runtime.get("args") or ()),
    )


def _primary_result(primary: Dict[str, Any], routing_reason: str = "primary_default") -> Dict[str, Any]:
    runtime = {
        "api_key": primary.get("api_key"),
        "base_url": primary.get("base_url"),
        "provider": primary.get("provider"),
        "api_mode": primary.get("api_mode"),
        "command": primary.get("command"),
        "args": list(primary.get("args") or []),
        "credential_pool": primary.get("credential_pool"),
    }
    return {
        "model": primary.get("model"),
        "runtime": runtime,
        "reasoning_config": primary.get("reasoning_config"),
        "routing_reason": routing_reason,
        "label": None,
        "signature": _route_signature(primary.get("model"), runtime),
    }


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route for non-complex turns.

    Preferred behavior:
    - default to the cheaper model for ordinary conversation/admin/meta turns
    - keep coding, debugging, deployment, research, and explicit planning work
      on the primary stronger model
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    cheap_model = _normalize_cheap_model_config(cfg)
    if cheap_model is None:
        return None

    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 600)
    max_words = _coerce_int(cfg.get("max_simple_words"), 96)

    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None
    if text.count("\n") > 2:
        return None
    if "```" in text or "`" in text:
        return None
    if _URL_RE.search(text):
        return None
    if _looks_like_complex_task(text):
        return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "cheap_default_non_complex_turn"
    return route


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn."""
    route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        return _primary_result(primary, routing_reason="primary_default")

    from hermes_cli.runtime_provider import resolve_runtime_provider

    reasoning_config = _route_reasoning_config(route, primary)
    route_provider = str(route.get("provider") or "").strip()
    route_base_url = route.get("base_url")
    route_api_key_env = str(route.get("api_key_env") or "").strip()
    explicit_api_key = os.getenv(route_api_key_env) if route_api_key_env else None

    if (
        route_provider
        and route_provider == str(primary.get("provider") or "").strip()
        and not route_base_url
        and not route_api_key_env
    ):
        runtime = {
            "api_key": primary.get("api_key"),
            "base_url": primary.get("base_url"),
            "provider": primary.get("provider"),
            "api_mode": primary.get("api_mode"),
            "command": primary.get("command"),
            "args": list(primary.get("args") or []),
            "credential_pool": primary.get("credential_pool"),
        }
    else:
        try:
            runtime = resolve_runtime_provider(
                requested=route_provider,
                explicit_api_key=explicit_api_key,
                explicit_base_url=route_base_url,
            )
        except Exception:
            return _primary_result(primary, routing_reason="primary_fallback")
        runtime = {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
        }

    return {
        "model": route.get("model"),
        "runtime": runtime,
        "reasoning_config": reasoning_config,
        "routing_reason": route.get("routing_reason") or "cheap_model_route",
        "label": f"smart route → {route.get('model')} ({runtime.get('provider')})",
        "signature": _route_signature(route.get("model"), runtime),
    }
