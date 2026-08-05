"""Translation helpers between Dana HostEvents and ACP SessionUpdates.

Dana's core session layer speaks :class:`HostEvent`; ACP speaks JSON-RPC
``session_update`` notifications with typed update chunks. This module is the
ONLY place where the two meet, keeping ACP types out of STAR core.

Chunk semantics: every ``ASSISTANT_CONTENT_CHUNK`` and ``USER_MESSAGE`` event
maps to an ACP delta (the client accumulates). ``ASSISTANT_CONTENT_FINAL`` is
NOT re-sent as a delta because the individual chunks already carried the text;
sending the full text again would duplicate it on the client side. Lifecycle
events (``TURN_*``, ``SESSION_*``) have no ACP update equivalent in D1 — the
``PromptResponse`` / ``LoadSessionResponse`` itself signals completion.

D2 adds tool lifecycle events: thought, tool-call, tool-update, result, and
cancellation states. These are translated to ACP ``agent_thought_chunk``,
``tool_call``, and ``tool_call_update`` notifications per ADR-013.

D6 adds multimodal content block translation: ``session/prompt`` content
blocks (text, image, embedded_resource, file_resource) are converted to
normalized dicts for Dana's core layer, and host events carrying multimodal
content are translated back to ACP ``user_message_chunk`` / ``agent_message_chunk``
updates with the appropriate content block types.
"""

from __future__ import annotations

from typing import Any

from acp.helpers import (
    start_tool_call,
    update_agent_message_text,
    update_agent_thought_text,
    update_tool_call,
    update_user_message_text,
)

from dana.core.session.projections.host_events import HostEvent, HostEventType


def host_event_to_acp_update(event: HostEvent) -> Any:
    """Translate a :class:`HostEvent` to an ACP SessionUpdate chunk, or ``None``.

    Returns ``None`` for events with no ACP equivalent (lifecycle events,
    content-final). Text-bearing events become delta chunks. Tool lifecycle
    events become ``tool_call`` or ``tool_call_update`` notifications.
    """
    # --- D1: Text-bearing events (with D6 multimodal content block support) ---
    if event.event_type is HostEventType.USER_MESSAGE:
        content_blocks = event.metadata.get("content_blocks")
        if content_blocks and isinstance(content_blocks, list):
            return _multimodal_user_message_to_acp(event)
        return update_user_message_text(event.text or "")
    if event.event_type is HostEventType.ASSISTANT_CONTENT_CHUNK:
        content_blocks = event.metadata.get("content_blocks")
        if content_blocks and isinstance(content_blocks, list):
            return _multimodal_agent_chunk_to_acp(event)
        return update_agent_message_text(event.text or "")
    # ASSISTANT_CONTENT_FINAL: already streamed via chunks — skip to avoid duplication.
    # TURN_*, SESSION_*: no ACP update in D1; the response signals completion.

    # --- D2: Agent thought ---
    if event.event_type is HostEventType.THOUGHT:
        return update_agent_thought_text(event.text or "")

    # --- D2: Tool lifecycle ---
    if event.event_type is HostEventType.TOOL_REQUESTED:
        return _tool_requested_to_acp(event)
    if event.event_type is HostEventType.TOOL_AUTHORIZED_OR_DENIED:
        return _tool_authorized_or_denied_to_acp(event)
    if event.event_type is HostEventType.TOOL_STARTED:
        return _tool_started_to_acp(event)
    if event.event_type is HostEventType.TOOL_PROGRESS:
        return _tool_progress_to_acp(event)
    if event.event_type is HostEventType.TOOL_CANCELLATION_REQUESTED:
        return _tool_cancellation_requested_to_acp(event)
    if event.event_type in (
        HostEventType.TOOL_RESULT,
        HostEventType.TOOL_FAILURE,
        HostEventType.TOOL_ACKNOWLEDGED,
        HostEventType.TOOL_TIMED_OUT,
        HostEventType.TOOL_EFFECT_UNKNOWN,
    ):
        return _tool_terminal_to_acp(event)

    return None


# ---------------------------------------------------------------------------
# D6: Multimodal content block translation helpers
# ---------------------------------------------------------------------------


def _multimodal_user_message_to_acp(event: HostEvent) -> Any:
    """Translate a USER_MESSAGE event with multimodal content blocks to ACP.

    The event's metadata carries ``content_blocks`` as a list of normalized
    block dicts. Each block is converted to the corresponding ACP content type:
    text → TextContentBlock, image → ImageContentBlock, embedded_resource →
    EmbeddedResourceContentBlock, file_resource → EmbeddedResourceContentBlock.
    """
    from acp.helpers import update_user_message

    content_blocks = event.metadata.get("content_blocks", [])
    acp_blocks = [_normalized_block_to_acp(b) for b in content_blocks if isinstance(b, dict)]
    # Use the first block as the primary content for the ACP update
    if acp_blocks:
        return update_user_message(acp_blocks[0])
    return update_user_message_text(event.text or "")


def _multimodal_agent_chunk_to_acp(event: HostEvent) -> Any:
    """Translate an ASSISTANT_CONTENT_CHUNK event with multimodal blocks to ACP."""
    from acp.helpers import update_agent_message

    content_blocks = event.metadata.get("content_blocks", [])
    acp_blocks = [_normalized_block_to_acp(b) for b in content_blocks if isinstance(b, dict)]
    if acp_blocks:
        return update_agent_message(acp_blocks[0])
    return update_agent_message_text(event.text or "")


def _normalized_block_to_acp(block: dict) -> Any:
    """Convert a normalized content block dict to an ACP content block.

    Normalized blocks have the shape produced by ContentNormalizer:
    - text: {"type": "text", "text": "..."}
    - image: {"type": "image", "media_type": "...", "content": b"..." or "data": "..."}
    - embedded_resource: {"type": "embedded_resource", "media_type": "...", "content": b"..."}
    - file_resource: {"type": "file_resource", "media_type": "...", "content": b"..."}

    Returns the appropriate ACP Pydantic model.
    """
    from acp.helpers import embedded_blob_resource, image_block, resource_block, text_block

    block_type = block.get("type", "")
    if block_type == "text":
        return text_block(text=block.get("text", ""))

    if block_type == "image":
        data = block.get("data") or block.get("content", b"")
        if isinstance(data, bytes):
            import base64

            data = base64.b64encode(data).decode("utf-8")
        return image_block(
            data=data,
            mime_type=block.get("media_type", "image/png"),
        )

    if block_type in ("embedded_resource", "file_resource"):
        data = block.get("data") or block.get("content", b"")
        if isinstance(data, bytes):
            import base64

            data = base64.b64encode(data).decode("utf-8")
        uri = block.get("artifact_uri") or block.get("uri", f"dana://{block.get('sha256', 'unknown')}")
        resource = embedded_blob_resource(
            uri=uri,
            blob=data,
            mime_type=block.get("media_type"),
        )
        return resource_block(resource=resource)

    return text_block(text=f"[{block_type} content]")


# ---------------------------------------------------------------------------
# ACP content block → normalized block conversion (for session/prompt input)
# ---------------------------------------------------------------------------


def acp_content_to_normalized_block(content: Any) -> dict:
    """Convert an ACP content block (Pydantic model or dict) to a normalized block dict.

    Handles the ACP content types that ``session/prompt`` can carry:
    - TextContentBlock (type="text") → {"type": "text", "text": "..."}
    - ImageContentBlock (type="image") → {"type": "image", "media_type": "...", "data": b"..."}
    - EmbeddedResourceContentBlock (type="resource") → {"type": "embedded_resource", ...}
    - ResourceContentBlock (type="resource_link") → {"type": "file_resource", ...}
    """
    if isinstance(content, dict):
        return _acp_dict_to_normalized(content)

    # Pydantic model
    content_type = getattr(content, "type", "")
    if content_type == "text":
        return {"type": "text", "text": getattr(content, "text", "")}
    if content_type == "image":
        return {
            "type": "image",
            "media_type": getattr(content, "mime_type", "image/png"),
            "data": getattr(content, "data", b""),
        }
    if content_type == "resource":
        resource = getattr(content, "resource", None)
        if resource is not None:
            return _acp_resource_to_normalized(resource)
    if content_type == "resource_link":
        return {
            "type": "file_resource",
            "uri": getattr(content, "uri", ""),
            "media_type": getattr(content, "mime_type", "application/octet-stream"),
        }
    return {"type": "text", "text": str(content)}


def _acp_dict_to_normalized(block: dict) -> dict:
    """Convert an ACP content block dict to a normalized block dict."""
    block_type = block.get("type", "")
    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "image":
        return {
            "type": "image",
            "media_type": block.get("mime_type", "image/png"),
            "data": block.get("data", b""),
        }
    if block_type == "resource":
        resource = block.get("resource", {})
        if isinstance(resource, dict):
            return _acp_resource_to_normalized(resource)
    if block_type == "resource_link":
        return {
            "type": "file_resource",
            "uri": block.get("uri", ""),
            "media_type": block.get("mime_type", "application/octet-stream"),
        }
    return {"type": "text", "text": str(block)}


def _acp_resource_to_normalized(resource: Any) -> dict:
    """Convert an ACP resource (TextResourceContents or BlobResourceContents) to a normalized block dict."""
    if isinstance(resource, dict):
        uri = resource.get("uri", "")
        mime_type = resource.get("mime_type") or resource.get("mimeType", "application/octet-stream")
        blob = resource.get("blob")
        if blob is not None:
            import base64

            try:
                data = base64.b64decode(blob)
            except Exception:
                data = blob.encode("utf-8")
            return {
                "type": "embedded_resource",
                "media_type": mime_type,
                "data": data,
                "uri": uri,
            }
        text = resource.get("text", "")
        return {
            "type": "embedded_resource",
            "media_type": mime_type,
            "data": text.encode("utf-8") if isinstance(text, str) else text,
            "uri": uri,
        }
    # Pydantic model
    uri = getattr(resource, "uri", "")
    mime_type = getattr(resource, "mime_type", "application/octet-stream")
    blob = getattr(resource, "blob", None)
    if blob is not None:
        import base64

        try:
            data = base64.b64decode(blob)
        except Exception:
            data = blob.encode("utf-8")
        return {
            "type": "embedded_resource",
            "media_type": mime_type,
            "data": data,
            "uri": uri,
        }
    text = getattr(resource, "text", "")
    return {
        "type": "embedded_resource",
        "media_type": mime_type,
        "data": text.encode("utf-8") if isinstance(text, str) else text,
        "uri": uri,
    }


# ---------------------------------------------------------------------------
# Tool lifecycle translation helpers
# ---------------------------------------------------------------------------


def _tool_requested_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_REQUESTED event to an ACP ``tool_call`` start notification.

    The ``tool_call`` notification carries the tool's identity, kind, and
    pending status. The client uses this to display a new tool card.
    """
    meta = event.metadata
    tool_name = meta.get("tool_name", "")
    tool_call_id = meta.get("tool_call_id", "")
    kind = meta.get("kind")
    return start_tool_call(
        tool_call_id=tool_call_id,
        title=tool_name,
        kind=kind,
        status="pending",
        raw_input=meta.get("raw_input"),
    )


def _tool_authorized_or_denied_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_AUTHORIZED_OR_DENIED event to an ACP tool_call_update.

    If the tool was denied, the status is ``failed`` with an error message.
    If authorized, the status remains ``pending`` (the TOOL_STARTED event
    will advance it to ``in_progress``).
    """
    meta = event.metadata
    tool_call_id = meta.get("tool_call_id", "")
    authorized = meta.get("authorized", True)
    if not authorized:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            raw_output={"error": meta.get("reason", "Permission denied")},
        )
    return update_tool_call(
        tool_call_id=tool_call_id,
        status="pending",
    )


def _tool_started_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_STARTED event to an ACP tool_call_update with in_progress status."""
    meta = event.metadata
    return update_tool_call(
        tool_call_id=meta.get("tool_call_id", ""),
        status="in_progress",
    )


def _tool_progress_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_PROGRESS event to an ACP tool_call_update with progress content."""
    meta = event.metadata
    return update_tool_call(
        tool_call_id=meta.get("tool_call_id", ""),
        status="in_progress",
        raw_output=meta.get("progress"),
    )


def _tool_cancellation_requested_to_acp(event: HostEvent) -> Any:
    """Translate a TOOL_CANCELLATION_REQUESTED event to an ACP tool_call_update.

    The tool call is being cancelled. The terminal outcome (acknowledged,
    timed-out, effect-unknown) will follow as a separate terminal event.
    """
    meta = event.metadata
    return update_tool_call(
        tool_call_id=meta.get("tool_call_id", ""),
        status="in_progress",
        raw_output={"cancellation": "requested"},
    )


def _tool_terminal_to_acp(event: HostEvent) -> Any:
    """Translate a terminal tool event to an ACP tool_call_update.

    Maps the five terminal outcomes to ACP status:
    - TOOL_RESULT → completed
    - TOOL_FAILURE → failed
    - TOOL_ACKNOWLEDGED → completed (cancellation acknowledged)
    - TOOL_TIMED_OUT → failed (cancellation timed out)
    - TOOL_EFFECT_UNKNOWN → failed (effect unknown)
    """
    meta = event.metadata
    tool_call_id = meta.get("tool_call_id", "")

    if event.event_type is HostEventType.TOOL_RESULT:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="completed",
            raw_output=meta.get("result"),
        )

    if event.event_type is HostEventType.TOOL_FAILURE:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            raw_output={"error": meta.get("error", "Tool execution failed")},
        )

    if event.event_type is HostEventType.TOOL_ACKNOWLEDGED:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="completed",
            raw_output={"cancellation": "acknowledged"},
        )

    if event.event_type is HostEventType.TOOL_TIMED_OUT:
        return update_tool_call(
            tool_call_id=tool_call_id,
            status="failed",
            raw_output={"cancellation": "timed_out"},
        )

    # TOOL_EFFECT_UNKNOWN
    return update_tool_call(
        tool_call_id=tool_call_id,
        status="failed",
        raw_output={"cancellation": "effect_unknown"},
    )
