"""Trace payload sanitizers for observability backends."""

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, cast


def safe_trace_inputs(inputs: dict) -> dict:
    """Return a JSON-safe trace input payload without recursive Dana objects."""
    return cast("dict", safe_trace_value(inputs))


def safe_trace_outputs(outputs: Any) -> dict:
    """Return a JSON-safe trace output payload without recursive Dana objects."""
    return {"output": safe_trace_value(outputs)}


def safe_trace_value(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Best-effort JSON-safe value for tracing.

    Dana often passes live agents, timelines, and runtime objects through
    observable methods; those graphs can be cyclic. Keep traces useful, but
    never recurse through the live object graph.
    """
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value

    obj_id = id(value)
    if obj_id in _seen:
        return f"<recursion:{value.__class__.__name__}>"

    if _is_dana_runtime_object(value):
        return _summarize_dana_object(value)

    if _depth >= 4:
        return _summarize_object(value)

    if isinstance(value, dict):
        _seen.add(obj_id)
        try:
            return {
                str(safe_trace_value(k, _seen=_seen, _depth=_depth + 1)): safe_trace_value(
                    v,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for k, v in value.items()
            }
        finally:
            _seen.discard(obj_id)
    if isinstance(value, list | tuple | set | frozenset):
        _seen.add(obj_id)
        try:
            return [safe_trace_value(item, _seen=_seen, _depth=_depth + 1) for item in list(value)[:50]]
        finally:
            _seen.discard(obj_id)
    if is_dataclass(value) and not isinstance(value, type):
        _seen.add(obj_id)
        try:
            return {
                field.name: safe_trace_value(getattr(value, field.name), _seen=_seen, _depth=_depth + 1)
                for field in fields(value)
                if field.repr
            }
        finally:
            _seen.discard(obj_id)

    return _summarize_object(value)


def _is_dana_runtime_object(value: Any) -> bool:
    module = value.__class__.__module__
    name = value.__class__.__name__
    return module.startswith("dana.core.agent") or module.startswith("dana.core.runtime") or name.endswith("Timeline")


def _summarize_dana_object(value: Any) -> dict:
    summary = _summarize_object(value)
    timeline = getattr(value, "timeline", None)
    if isinstance(timeline, list):
        summary["entry_count"] = len(timeline)
    for attr in ("max_context_tokens", "agent_type", "object_id", "session_id"):
        try:
            attr_value = getattr(value, attr)
        except Exception:
            continue
        if isinstance(attr_value, str | int | float | bool):
            summary[attr] = attr_value
    return summary


def _summarize_object(value: Any) -> dict:
    return {
        "__class__": value.__class__.__name__,
        "__module__": value.__class__.__module__,
    }
