"""D8: Claude Code skills registration is opt-in (DANA_CLAUDE_SKILLS=1).

Default agent construction must perform NO scan of ~/.claude/skills — verified
by mocking the ClaudeCodeSkills loader and asserting zero calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dana.core.agent.star_agent import STARAgent


def _make_agent():
    """Create a minimal STARAgent without registry or LLM side-effects."""
    return STARAgent(
        agent_type="test-agent",
        auto_register=False,
        enable_web_search=False,
        enable_code_execution=False,
        enable_assistant=False,
        compress_timeline=False,
    )


def test_default_construction_skills_loader_never_called(monkeypatch):
    loader = MagicMock()
    monkeypatch.delenv("DANA_CLAUDE_SKILLS", raising=False)
    monkeypatch.setattr("dana.core.skills.ClaudeCodeSkills", loader)

    with patch("dana.core.agent.star_agent.LLM"):
        agent = _make_agent()

    loader.assert_not_called()
    assert not any(r.resource_type == "claude-skills" for r in agent.available_resources)


def test_opt_in_env_registers_skills(monkeypatch):
    loader = MagicMock()
    instance = MagicMock()
    instance.enabled = True
    instance.resource_type = "claude-skills"
    loader.return_value = instance

    monkeypatch.setenv("DANA_CLAUDE_SKILLS", "1")
    monkeypatch.setattr("dana.core.skills.ClaudeCodeSkills", loader)

    with patch("dana.core.agent.star_agent.LLM"):
        agent = _make_agent()

    loader.assert_called_once()
    assert any(r.resource_type == "claude-skills" for r in agent.available_resources)


def test_garbage_env_skips_registration(monkeypatch):
    loader = MagicMock()
    monkeypatch.setenv("DANA_CLAUDE_SKILLS", "banana")
    monkeypatch.setattr("dana.core.skills.ClaudeCodeSkills", loader)

    with patch("dana.core.agent.star_agent.LLM"):
        _make_agent()

    loader.assert_not_called()
