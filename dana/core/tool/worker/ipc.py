"""IPC protocol for isolated worker communication (D2).

Defines the request/response message format exchanged over stdin/stdout
pipes between the parent process and the isolated worker subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any


class WorkerStatus(Enum):
    """Status of a worker request after execution."""

    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    CRASHED = "crashed"


@dataclass
class WorkerRequest:
    """A request sent to the isolated worker process.

    ``request_id``   — unique identifier for this request (echoed back).
    ``module``       — Python module path to import (e.g. ``"os"``).
    ``callable``     — callable name within the module, e.g. ``"path.join"``.
    ``args``         — positional arguments (JSON-serializable).
    ``kwargs``       — keyword arguments (JSON-serializable).
    ``timeout_ms``   — max wall-clock time for execution (0 = no timeout).
    """

    request_id: str
    module: str
    callable: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 0


@dataclass
class WorkerResponse:
    """A response from the isolated worker process.

    ``request_id`` — echoes the request's ``request_id``.
    ``status``     — one of SUCCESS, ERROR, CANCELLED, CRASHED.
    ``result``     — the return value (on SUCCESS) or error info.
    """

    request_id: str
    status: WorkerStatus
    result: Any = None


def encode_request(request: WorkerRequest) -> str:
    """Serialize a WorkerRequest to a JSON line."""
    return json.dumps(
        {
            "type": "request",
            "request_id": request.request_id,
            "module": request.module,
            "callable": request.callable,
            "args": request.args,
            "kwargs": request.kwargs,
            "timeout_ms": request.timeout_ms,
        }
    )


def decode_request(line: str) -> WorkerRequest:
    """Deserialize a JSON line to a WorkerRequest."""
    data = json.loads(line)
    if data.get("type") != "request":
        raise ValueError(f"Expected request type, got: {data.get('type')}")
    return WorkerRequest(
        request_id=data["request_id"],
        module=data["module"],
        callable=data["callable"],
        args=tuple(data.get("args", [])),
        kwargs=data.get("kwargs", {}),
        timeout_ms=data.get("timeout_ms", 0),
    )


def encode_response(response: WorkerResponse) -> str:
    """Serialize a WorkerResponse to a JSON line."""
    return json.dumps(
        {
            "type": "response",
            "request_id": response.request_id,
            "status": response.status.value,
            "result": response.result,
        }
    )


def decode_response(line: str) -> WorkerResponse:
    """Deserialize a JSON line to a WorkerResponse."""
    data = json.loads(line)
    if data.get("type") != "response":
        raise ValueError(f"Expected response type, got: {data.get('type')}")
    return WorkerResponse(
        request_id=data["request_id"],
        status=WorkerStatus(data["status"]),
        result=data.get("result"),
    )
