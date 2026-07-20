"""Small synchronous client shared by routing agents.

The module contains provider-specific I/O only. It has no routing rules,
fallback policy, background worker, or mutable agent state.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from config import ACTIVE_AGENT_MODEL, MODEL_CONFIGS, verbosity_level


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

load_dotenv(ROOT_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(ROOT_DIR / "environment")
load_dotenv(PROJECT_DIR / "environment")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} was not found. Put it in your .env file.")
    return value


def parse_json_response(text: str) -> Optional[dict]:
    """Parse plain JSON, fenced JSON, or a JSON object surrounded by prose."""
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None


def _content_to_text(content) -> str:
    """Normalize provider text blocks before JSON parsing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content or "")


def call_llm(
    system_prompt: str,
    user_payload: dict,
    model_key: str = ACTIVE_AGENT_MODEL,
    timeout: float = 15.0,
    response_schema: Optional[dict] = None,
) -> tuple[Optional[dict], dict]:
    """Call one configured model synchronously and return parsed JSON + metrics."""
    cfg = MODEL_CONFIGS[model_key]
    provider = cfg["provider"]
    model = cfg["model"]
    temperature = cfg.get("temperature", 0.0)
    max_tokens = cfg.get("max_tokens", 300)
    user_content = json.dumps(user_payload, separators=(",", ":"))
    started_at = time.perf_counter()

    metrics = {
        "provider": provider,
        "model": model,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "status": "ok",
    }
    raw_text = ""

    try:
        if provider == "ollama":
            import requests

            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                    "keep_alive": cfg.get("keep_alive", "5m"),
                    "options": {
                        "temperature": temperature,
                        "num_ctx": cfg.get("num_ctx", 2048),
                    },
                },
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            metrics["input_tokens"] = body.get("prompt_eval_count")
            metrics["output_tokens"] = body.get("eval_count")
            if metrics["input_tokens"] is not None and metrics["output_tokens"] is not None:
                metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
            raw_text = _content_to_text(body["message"]["content"])
            result = parse_json_response(raw_text)

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            google_args = dict(
                model=model,
                google_api_key=_required_env("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                timeout=timeout,
                max_retries=0,
            )
            if "thinking_budget" in cfg:
                google_args["thinking_budget"] = cfg["thinking_budget"]
            llm = ChatGoogleGenerativeAI(**google_args).bind(
                response_mime_type="application/json"
            )
            response = llm.invoke([
                ("system", system_prompt),
                ("human", user_content),
            ])
            usage = getattr(response, "usage_metadata", None) or {}
            metrics.update({
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            })
            raw_text = _content_to_text(getattr(response, "content", str(response)))
            result = parse_json_response(raw_text)

        else:
            providers = {
                "openai": ("OPENAI_API_KEY", None),
                "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
                "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
                "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
                "requesty": ("REQUESTY_API_KEY", "https://router.requesty.ai/v1"),
            }
            if provider not in providers:
                raise ValueError(f"Unknown provider: {provider}")

            default_key, default_url = providers[provider]
            client_args = {
                "api_key": _required_env(cfg.get("api_key_env", default_key)),
                "max_retries": 0,
            }
            base_url = cfg.get("base_url", default_url)
            if base_url:
                client_args["base_url"] = base_url
            if provider == "openrouter":
                client_args["default_headers"] = {
                    "HTTP-Referer": cfg.get("site_url", "http://localhost"),
                    "X-OpenRouter-Title": cfg.get(
                        "app_name", "Agentic Network Management Project"
                    ),
                }

            if provider == "requesty" and model.startswith("policy/"):
                raise ValueError(
                    "Requesty policy models are disabled because they may "
                    "route to fallback models. Configure an exact "
                    "provider/model identifier instead."
                )

            extra_body = None
            if provider == "openrouter":
                # A failed provider/model request must be visible to the agent;
                # OpenRouter must not silently select another provider.
                extra_body = {"provider": {"allow_fallbacks": False}}
            elif provider == "requesty":
                # Exact model IDs avoid Requesty routing policies. Explicitly
                # bypass caching so every agent decision is a fresh model call.
                extra_body = {"requesty": {"auto_cache": False}}

            request_args = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if provider == "requesty":
                if response_schema is not None:
                    # Constrained decoding prevents models such as Claude from
                    # spending their whole output budget on explanatory prose.
                    request_args["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "routing_policy",
                            "strict": True,
                            "schema": response_schema,
                        },
                    }
                else:
                    request_args["response_format"] = {"type": "json_object"}
            if extra_body is not None:
                request_args["extra_body"] = extra_body

            response = OpenAI(**client_args).chat.completions.create(
                **request_args
            )
            choice = response.choices[0]
            metrics["finish_reason"] = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            if usage is not None:
                completion_details = getattr(
                    usage, "completion_tokens_details", None
                )
                metrics.update({
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "cost_usd": getattr(usage, "cost", None),
                    "reasoning_tokens": getattr(
                        completion_details, "reasoning_tokens", None
                    ),
                })
            raw_text = _content_to_text(choice.message.content)
            result = parse_json_response(raw_text)

        metrics["response_characters"] = len(raw_text)
        metrics["response_preview"] = raw_text[:500]
        if result is None:
            metrics["status"] = "invalid_json"
        metrics["inference_time_ms"] = (time.perf_counter() - started_at) * 1000
        return result, metrics

    except Exception as exc:
        if verbosity_level >= 1:
            print(f"[RoutingAgent] LLM error: provider={provider}, model={model}: {exc}")
        metrics.update({
            "status": "error",
            "error": str(exc),
            "inference_time_ms": (time.perf_counter() - started_at) * 1000,
        })
        return None, metrics
