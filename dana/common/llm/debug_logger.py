"""Opt-in, bounded LLM diagnostics containing metadata only.

Prompt text, model output, tool arguments, exception messages, credentials and
provider replay material are never serialized. Disabled logging does no disk IO.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .types import LLMMessage, LLMResponse


_MAX_LOG_BYTES = 4 * 1024 * 1024


class LLMDebugLogger:
    """Metadata diagnostics, enabled explicitly or by DANA_LLM_DEBUG_LOGGING."""

    def __init__(self):
        self.enabled = os.getenv("DANA_LLM_DEBUG_LOGGING", "").lower() in {"1", "true", "yes"}
        self._resolved = False
        self._log_dir: Path | None = None
        self._request_log: Path | None = None
        self._response_log: Path | None = None
        self._agent_log: Path | None = None
        self._error_log: Path | None = None
        self._write_lock = Lock()

    def _ensure_resolved(self) -> None:
        if not self.enabled or self._resolved:
            return
        self._resolved = True
        self._log_dir = self._resolve_log_dir()
        if self._log_dir is not None:
            self._request_log = self._log_dir / "llm_requests.jsonl"
            self._response_log = self._log_dir / "llm_responses.jsonl"
            self._agent_log = self._log_dir / "agent_interactions.jsonl"
            self._error_log = self._log_dir / "errors.jsonl"

    @property
    def log_dir(self) -> Path | None:
        self._ensure_resolved()
        return self._log_dir

    @property
    def request_log(self) -> Path | None:
        self._ensure_resolved()
        return self._request_log

    @property
    def response_log(self) -> Path | None:
        self._ensure_resolved()
        return self._response_log

    @property
    def agent_log(self) -> Path | None:
        self._ensure_resolved()
        return self._agent_log

    @property
    def error_log(self) -> Path | None:
        self._ensure_resolved()
        return self._error_log

    @staticmethod
    def _label(value: str | None) -> str | None:
        return value[:160] if value is not None else None

    def log_request(
        self, provider: str, model: str, messages: list[LLMMessage], agent_id: str | None = None, agent_type: str | None = None, **kwargs
    ) -> str:
        request_id = uuid4().hex
        if self.enabled:
            self._write_log_entry(
                self.request_log,
                {
                    "type": "request",
                    "request_id": request_id,
                    "provider": self._label(provider),
                    "model": self._label(model),
                    "agent_id": self._label(agent_id),
                    "agent_type": self._label(agent_type),
                    "message_count": len(messages),
                },
            )
        return request_id

    def log_response(
        self,
        request_id: str,
        response: LLMResponse,
        provider: str,
        model: str,
        agent_id: str | None = None,
        agent_type: str | None = None,
        duration_ms: float | None = None,
    ):
        if not self.enabled:
            return
        self._write_log_entry(
            self.response_log,
            {
                "type": "response",
                "request_id": self._label(request_id),
                "provider": self._label(provider),
                "model": self._label(response.model or model),
                "agent_id": self._label(agent_id),
                "agent_type": self._label(agent_type),
                "duration_ms": duration_ms,
                "content_length": len(response.content) if response.content else 0,
                "finish_reason": self._label(response.finish_reason),
                "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0,
            },
        )

    def log_full_response_content(self, request_id: str, content: str, provider: str, model: str):
        """Record only the length; full response payload logging is forbidden."""
        if self.enabled:
            self._write_log_entry(
                self.response_log,
                {
                    "type": "response_content_redacted",
                    "request_id": self._label(request_id),
                    "provider": self._label(provider),
                    "model": self._label(model),
                    "content_length": len(content) if content else 0,
                },
            )

    def log_agent_interaction(
        self,
        agent_id: str,
        agent_type: str,
        interaction_type: str,
        content: str,
        target_agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if self.enabled:
            self._write_log_entry(
                self.agent_log,
                {
                    "type": "agent_interaction",
                    "agent_id": self._label(agent_id),
                    "agent_type": self._label(agent_type),
                    "interaction_type": self._label(interaction_type),
                    "target_agent_id": self._label(target_agent_id),
                    "content_length": len(content) if content else 0,
                },
            )

    def log_error(
        self,
        error: Exception,
        context: str,
        request_id: str | None = None,
        agent_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ):
        if self.enabled:
            self._write_log_entry(
                self.error_log,
                {
                    "type": "error",
                    "request_id": self._label(request_id),
                    "agent_id": self._label(agent_id),
                    "provider": self._label(provider),
                    "model": self._label(model),
                    "context": self._label(context),
                    "error_type": type(error).__name__,
                },
            )

    def _write_log_entry(self, log_file: Path | None, entry: dict[str, Any]):
        if not self.enabled or log_file is None:
            return
        entry["timestamp"] = datetime.now().isoformat()
        with self._write_lock:
            try:
                encoded = (json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                if log_file.exists() and log_file.stat().st_size + len(encoded) > _MAX_LOG_BYTES:
                    log_file.replace(log_file.with_suffix(".jsonl.1"))
                with log_file.open("ab") as stream:
                    stream.write(encoded)
            except (OSError, TypeError, ValueError):
                # Optional diagnostics must not turn a successful request into a failure.
                self.enabled = False

    def get_log_stats(self) -> dict[str, Any]:
        if self.log_dir is None:
            return {"log_directory": None, "log_files": {}}
        files = {}
        for path in (self.request_log, self.response_log, self.agent_log, self.error_log):
            if path is not None:
                size = path.stat().st_size if path.exists() else 0
                line_count = 0
                if size:
                    with path.open(encoding="utf-8") as stream:
                        line_count = sum(1 for _ in stream)
                files[path.name] = {"size_bytes": size, "line_count": line_count}
        return {"log_directory": str(self.log_dir), "log_files": files}

    def cleanup_old_logs(self, days_to_keep: int = 7):
        if self.log_dir is None:
            return
        cutoff = datetime.now().timestamp() - days_to_keep * 24 * 60 * 60
        for path in self.log_dir.glob("*.jsonl*"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

    def _resolve_log_dir(self) -> Path | None:
        for candidate in (Path.home() / ".dana" / "logs", Path.cwd() / ".dana" / "logs"):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError:
                continue
        return None


_debug_logger: LLMDebugLogger | None = None


def get_debug_logger() -> LLMDebugLogger:
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = LLMDebugLogger()
    return _debug_logger


def enable_debug_logging():
    """Enable bounded, metadata-only diagnostics."""
    get_debug_logger().enabled = True


def disable_debug_logging():
    """Disable diagnostics, including an instance retained by an active caller."""
    get_debug_logger().enabled = False
