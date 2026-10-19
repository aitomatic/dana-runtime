"""Diagnostics never persist prompt, provider replay, or credential payloads."""

import json

from dana.common.llm import debug_logger as module
from dana.common.llm.types import LLMMessage, LLMResponse


def test_payloads_are_redacted_and_files_rotate(monkeypatch, tmp_path):
    monkeypatch.setenv("DANA_LLM_DEBUG_LOGGING", "true")
    monkeypatch.setattr(module.LLMDebugLogger, "_resolve_log_dir", lambda self: tmp_path)
    monkeypatch.setattr(module, "_MAX_LOG_BYTES", 1000)
    logger = module.LLMDebugLogger()
    secret = "synthetic-secret-prompt-and-replay-material"
    for _ in range(20):
        request_id = logger.log_request("custom", "fixture", [LLMMessage(content=secret, role="user")], api_key=secret)
        logger.log_response(request_id, LLMResponse(content=secret, model="fixture"), "custom", "fixture")
        logger.log_full_response_content(request_id, secret, "custom", "fixture")
        logger.log_agent_interaction("agent", "fixture", "query", secret, metadata={"encrypted_content": secret})
        logger.log_error(ValueError(secret), "chat_response", request_id=request_id)
    paths = tuple(tmp_path.iterdir())
    assert set(path.name for path in paths) == {
        f"{name}.jsonl{suffix}" for name in ("llm_requests", "llm_responses", "agent_interactions", "errors") for suffix in ("", ".1")
    }
    for path in paths:
        data = path.read_text()
        assert secret not in data
        assert path.stat().st_size <= 1000
        for line in data.splitlines():
            assert isinstance(json.loads(line)["type"], str)
    logger.enabled = False
    before = {path: path.read_bytes() for path in paths}
    logger.log_request("custom", "fixture", [LLMMessage(content=secret, role="user")])
    logger.log_error(ValueError(secret), "chat_response")
    assert {path: path.read_bytes() for path in paths} == before


def test_disabled_diagnostics_never_resolve_storage(monkeypatch):
    monkeypatch.delenv("DANA_LLM_DEBUG_LOGGING", raising=False)

    def unexpected_storage(self):
        raise AssertionError("disabled diagnostics touched storage")

    monkeypatch.setattr(module.LLMDebugLogger, "_resolve_log_dir", unexpected_storage)
    logger = module.LLMDebugLogger()
    request_id = logger.log_request("custom", "fixture", [])
    logger.log_response(request_id, LLMResponse(content="output", model="fixture"), "custom", "fixture")
    logger.log_full_response_content(request_id, "output", "custom", "fixture")
    logger.log_agent_interaction("agent", "fixture", "query", "prompt")
    logger.log_error(ValueError("error"), "chat_response")
    assert logger.get_log_stats() == {"log_directory": None, "log_files": {}}
