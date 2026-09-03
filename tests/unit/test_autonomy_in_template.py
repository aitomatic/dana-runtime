"""Test that autonomy instructions are properly placed in the system prompt template."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

from dana.core.agent.star_agent import STARAgent
from dana.core.knowledge.prompts.codecs import CSXMLCodec, KLXMLCodec
from dana.core.prompt.prompt_api import TEMPLATE_SYSTEM_PROMPT


class TestAutonomyInTemplate:
    """Test that autonomy is in the template, not the codec."""

    def test_template_contains_autonomous_operation_section(self):
        """Verify the template has the <output_format> section."""
        assert "<output_format>" in TEMPLATE_SYSTEM_PROMPT
        assert "</output_format>" in TEMPLATE_SYSTEM_PROMPT

    def test_template_contains_key_autonomy_rules(self):
        """Verify the template contains key autonomy rules."""
        # Check for critical autonomy sections (current template structure)
        assert "<autonomy>" in TEMPLATE_SYSTEM_PROMPT
        assert "</autonomy>" in TEMPLATE_SYSTEM_PROMPT
        assert "<decision_framework>" in TEMPLATE_SYSTEM_PROMPT
        assert "{{tool_instruction_prompt}}" in TEMPLATE_SYSTEM_PROMPT  # Codec-specific output format

    def test_template_no_todo_resource_reference(self):
        """Verify the template no longer mentions the todo-resource."""
        assert "todo-resource" not in TEMPLATE_SYSTEM_PROMPT

    def test_csxml_codec_has_no_autonomy(self):
        """Verify CSXMLCodec only has format rules, no autonomy."""
        instructions = CSXMLCodec.get_instruction()

        # Should NOT contain autonomy-related phrases
        assert "AUTONOMOUS" not in instructions
        assert "multi-step" not in instructions.lower()
        assert "After EACH tool result" not in instructions

        # Should contain format-related phrases
        assert "<thinking>" in instructions
        assert "<response>" in instructions
        assert "<function_call>" in instructions
        assert "FORMAT RULES" in instructions

    def test_klxml_codec_has_no_autonomy(self):
        """Verify KLXMLCodec only has format rules, no autonomy."""
        instructions = KLXMLCodec.get_instruction()

        # Should NOT contain autonomy-related phrases
        assert "AUTONOMOUS" not in instructions
        assert "multi-step" not in instructions.lower()
        assert "After EACH tool result" not in instructions

        # Should contain format-related phrases
        assert "<thinking>" in instructions
        assert "<response>" in instructions
        assert "FORMAT RULES" in instructions


class TestAgentSystemPromptIncludesAutonomy:
    """Test that actual agent system prompts include autonomy (JSON format via DefaultRuntime)."""

    def test_star_agent_system_prompt_has_autonomy(self):
        """Verify STARAgent system prompt includes autonomy instructions."""

        class TestAgent(STARAgent):
            """A test agent for verification."""

            def __init__(self):
                super().__init__(
                    agent_type="test",
                    agent_id="test-001",
                    auto_register=False,
                    enable_web_search=False,
                    enable_skills=False,
                    enable_code_execution=False,
                )

        agent = TestAgent()
        system_prompt = agent.system_prompt

        assert "<autonomy>" in system_prompt
        assert "<output_format>" in system_prompt

    def test_codec_runtime_full_prompt_override_refreshes_actual_llm_prompt(self):
        agent = STARAgent(
            agent_type="custom",
            agent_id="custom-prompt-override",
            auto_register=False,
            enable_assistant=False,
            enable_web_search=False,
            enable_skills=False,
            enable_code_execution=False,
        )
        runtime = agent._runtime
        prompt_api = runtime._get_prompt_api(agent)
        prompt_api._store = Mock()

        assert prompt_api._repository_factory is agent._repository_factory

        agent.override_system_prompt_template("first prompt")
        assert agent.system_prompt == "first prompt"

        agent.override_system_prompt_template("second prompt")
        runtime._build_native_tools_if_supported = Mock()
        runtime._get_runtime_context = Mock(return_value={})
        messages = runtime.build_prompt(agent, agent._timeline)

        assert agent.system_prompt == "second prompt"
        assert messages[0].content == "second prompt"
        prompt_api._store.get_active.assert_not_called()
        prompt_api._store.create_snapshot.assert_not_called()

    def test_shared_codec_runtime_keeps_overrides_agent_scoped(self):
        first = STARAgent(
            agent_type="first",
            agent_id="first-codec",
            auto_register=False,
            enable_assistant=False,
            enable_web_search=False,
            enable_skills=False,
            enable_code_execution=False,
        )
        runtime = first._runtime
        second = STARAgent(
            agent_type="second",
            agent_id="second-codec",
            runtime=runtime,
            auto_register=False,
            enable_assistant=False,
            enable_web_search=False,
            enable_skills=False,
            enable_code_execution=False,
        )
        first_store = Mock()
        first_store.get_active.return_value = None
        second_store = Mock()
        second_store.get_active.return_value = None
        runtime._get_prompt_api(first)._store = first_store
        runtime._get_prompt_api(second)._store = second_store

        first.override_system_prompt_template("first codec override")

        assert first.system_prompt == "first codec override"
        assert second.system_prompt != "first codec override"

    def test_shared_codec_runtime_builds_concurrent_prompts_for_current_agent(self):
        first = STARAgent(
            agent_type="first",
            agent_id="first-concurrent-codec",
            auto_register=False,
            enable_assistant=False,
            enable_web_search=False,
            enable_skills=False,
            enable_code_execution=False,
        )
        runtime = first._runtime
        second = STARAgent(
            agent_type="second",
            agent_id="second-concurrent-codec",
            runtime=runtime,
            auto_register=False,
            enable_assistant=False,
            enable_web_search=False,
            enable_skills=False,
            enable_code_execution=False,
        )
        runtime._get_prompt_api(first)._store = Mock()
        runtime._get_prompt_api(second)._store = Mock()
        first.override_system_prompt_template("first concurrent override")
        second.override_system_prompt_template("second concurrent override")

        barrier = Barrier(2)

        def wait_for_both_agents() -> dict:
            barrier.wait()
            return {}

        runtime._build_native_tools_if_supported = Mock()
        runtime._build_tool_name_registry = Mock()
        runtime._get_runtime_context = wait_for_both_agents

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(runtime.build_prompt, first, first._timeline)
            second_future = executor.submit(runtime.build_prompt, second, second._timeline)

        assert first_future.result()[0].content == "first concurrent override"
        assert second_future.result()[0].content == "second concurrent override"

    def test_codec_getter_matches_persisted_prompt_used_for_llm_messages(self):
        agent = STARAgent(
            agent_type="persisted",
            agent_id="persisted-codec",
            auto_register=False,
            enable_assistant=False,
            enable_web_search=False,
            enable_skills=False,
            enable_code_execution=False,
        )
        runtime = agent._runtime
        prompt_api = runtime._get_prompt_api(agent)
        snapshot = Mock()
        snapshot.content = "persisted codec prompt"
        prompt_api._store = Mock()
        prompt_api._store.get_active.return_value = snapshot
        runtime._build_native_tools_if_supported = Mock()
        runtime._get_runtime_context = Mock(return_value={})

        messages = runtime.build_prompt(agent, agent._timeline)

        assert agent.system_prompt == "persisted codec prompt"
        assert messages[0].content == "persisted codec prompt"

    def test_star_agent_subclass_inherits_autonomy(self):
        """Verify subclasses of STARAgent inherit autonomy instructions."""

        class CustomAgent(STARAgent):
            """A custom agent that extends STARAgent."""

            def __init__(self):
                super().__init__(
                    agent_type="custom",
                    agent_id="custom-001",
                    auto_register=False,
                    enable_web_search=False,
                    enable_skills=False,
                    enable_code_execution=False,
                )

        agent = CustomAgent()
        system_prompt = agent.system_prompt

        assert "<autonomy>" in system_prompt
        assert "<output_format>" in system_prompt
