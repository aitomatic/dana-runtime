"""Shared normalized-block → ``TextBlock`` conversion for multimodal turns.

Per ADR-009 (Multimodal Content and Artifact References): host adapters build
*normalized* content block dicts (text / image / embedded_resource / file_resource)
and convert them to the ``TextBlock`` list + ``content_blocks`` payload that
``AgentSession.prompt`` consumes. This module owns that conversion so both
``dana-acp`` and ``dana-code`` share one faithful implementation.

Moved from ``dana.apps.acp.agent._normalized_blocks_to_text_blocks`` so the CLI
does not duplicate it.
"""

from __future__ import annotations

from dana.core.session.agent_session import TextBlock


def normalized_blocks_to_text_blocks(blocks: list[dict]) -> tuple[list[TextBlock], list[dict]]:
    """Convert normalized block dicts to a ``(TextBlock list, content_blocks payload)``.

    Text blocks become part of the joined prompt text. Multimodal blocks are
    serialized as text placeholders with their content preserved in the
    ``content_blocks`` payload (image/resource bytes are base64-encoded so the
    payload is JSON-serializable for journaling).

    Args:
        blocks: Normalized content block dicts (``type`` in
            ``{text, image, embedded_resource, file_resource}``).

    Returns:
        A ``(text_blocks, content_blocks_payload)`` pair. When there is no
        multimodal content, ``content_blocks_payload`` carries only the text
        blocks (mirroring the ACP behaviour).
    """
    text_parts: list[str] = []
    has_multimodal = any(b.get("type") != "text" for b in blocks)
    content_blocks_payload: list[dict] = []

    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            text_parts.append(text)
            content_blocks_payload.append(block)
        elif block_type == "image":
            media_type = block.get("media_type", "image/*")
            text_parts.append(f"[Image: {media_type}]")
            data = block.get("data", b"")
            if isinstance(data, bytes):
                import base64

                block["data"] = base64.b64encode(data).decode("utf-8")
            content_blocks_payload.append(block)
        elif block_type in ("embedded_resource", "file_resource"):
            media_type = block.get("media_type", "application/octet-stream")
            uri = block.get("uri", "")
            text_parts.append(f"[Resource: {media_type}]" if not uri else f"[Resource: {uri}]")
            data = block.get("data", b"")
            if isinstance(data, bytes):
                import base64

                block["data"] = base64.b64encode(data).decode("utf-8")
            content_blocks_payload.append(block)

    if not text_parts and not has_multimodal:
        return [TextBlock(text="")], content_blocks_payload

    text = " ".join(text_parts) if text_parts else "[multimodal content]"
    return [TextBlock(text=text)], content_blocks_payload
