"""Tests for ToDoResource.todo_write input handling.

Covers:
- Mapping input (the shape a decoded tool call produces) - unchanged behavior
- TodoItem model input (the type todo_write declares in its signature)
- Attribute-style items (dana.core.runtime.TodoItem dataclass)
- Validation errors and rendering are identical across all accepted shapes
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dana.core.resource.todo_resource import TodoItem, ToDoResource


@pytest.fixture
def todo_resource():
    """Create a ToDoResource instance."""
    return ToDoResource(resource_id="test-todo")


@dataclass
class AttrTodo:
    """Dataclass-style todo item, mirroring dana.core.runtime.TodoItem."""

    content: str
    status: str
    activeForm: str


# ============================================================================
# ACCEPTED INPUT SHAPES
# ============================================================================


class TestAcceptedInputShapes:
    """todo_write should accept every todo item shape used in the codebase."""

    @pytest.mark.asyncio
    async def test_mapping_input_renders_list(self, todo_resource):
        """Dict input (decoded tool-call JSON) renders the todo list."""
        result = await todo_resource.todo_write(todos=[{"content": "Run tests", "status": "pending", "activeForm": "Running tests"}])

        assert "Todo List:" in result
        assert "Run tests" in result
        assert "Progress: 0/1 completed, 0 in progress, 1 pending" in result

    @pytest.mark.asyncio
    async def test_todoitem_model_input_renders_list(self, todo_resource):
        """The TodoItem model declared by todo_write is accepted."""
        result = await todo_resource.todo_write(todos=[TodoItem(content="Run tests", status="pending", activeForm="Running tests")])

        assert "Todo List:" in result
        assert "Run tests" in result
        assert "Progress: 0/1 completed, 0 in progress, 1 pending" in result

    @pytest.mark.asyncio
    async def test_attribute_item_input_renders_list(self, todo_resource):
        """Dataclass-style items exposing the fields as attributes are accepted."""
        result = await todo_resource.todo_write(todos=[AttrTodo(content="Run tests", status="pending", activeForm="Running tests")])

        assert "Todo List:" in result
        assert "Run tests" in result

    @pytest.mark.asyncio
    async def test_shapes_render_identically(self, todo_resource):
        """The same todo expressed in each shape produces the same output."""
        fields = {"content": "Build project", "status": "in_progress", "activeForm": "Building project"}

        from_dict = await todo_resource.todo_write(todos=[dict(fields)])
        from_model = await todo_resource.todo_write(todos=[TodoItem(**fields)])
        from_attrs = await todo_resource.todo_write(todos=[AttrTodo(**fields)])

        assert from_dict == from_model == from_attrs

    @pytest.mark.asyncio
    async def test_mixed_shapes_in_one_call(self, todo_resource):
        """A list mixing shapes is normalized as a whole."""
        result = await todo_resource.todo_write(
            todos=[
                {"content": "First", "status": "completed", "activeForm": "Doing first"},
                TodoItem(content="Second", status="in_progress", activeForm="Doing second"),
                AttrTodo(content="Third", status="pending", activeForm="Doing third"),
            ]
        )

        assert "First" in result
        assert "Doing second" in result
        assert "Third" in result
        assert "Progress: 1/3 completed, 1 in progress, 1 pending" in result


# ============================================================================
# STORED STATE
# ============================================================================


class TestStoredTodos:
    """get_todos should return dicts regardless of the input shape."""

    @pytest.mark.asyncio
    async def test_model_input_is_stored_as_dict(self, todo_resource):
        """Model input is normalized before it is stored."""
        await todo_resource.todo_write(todos=[TodoItem(content="Run tests", status="pending", activeForm="Running tests")])

        stored = todo_resource.get_todos()

        assert stored == [{"content": "Run tests", "status": "pending", "activeForm": "Running tests"}]

    @pytest.mark.asyncio
    async def test_dict_input_is_stored_unchanged(self, todo_resource):
        """Dict input round-trips through get_todos unchanged."""
        todo = {"content": "Run tests", "status": "completed", "activeForm": "Running tests"}

        await todo_resource.todo_write(todos=[todo])

        assert todo_resource.get_todos() == [todo]


# ============================================================================
# VALIDATION
# ============================================================================


class TestValidation:
    """Validation messages should be the same for every accepted shape."""

    @pytest.mark.asyncio
    async def test_missing_field_in_mapping_reports_error(self, todo_resource):
        """A dict missing activeForm is reported, not rendered."""
        result = await todo_resource.todo_write(todos=[{"content": "Run tests", "status": "pending"}])

        assert result == "Error: Todo item 0 missing 'activeForm' field"

    @pytest.mark.asyncio
    async def test_missing_field_in_attribute_item_reports_error(self, todo_resource):
        """An attribute item missing activeForm is reported, not a TypeError."""

        @dataclass
        class PartialTodo:
            content: str
            status: str

        result = await todo_resource.todo_write(todos=[PartialTodo(content="Run tests", status="pending")])

        assert result == "Error: Todo item 0 missing 'activeForm' field"

    @pytest.mark.asyncio
    async def test_invalid_status_in_model_reports_error(self, todo_resource):
        """An invalid status on a model item is reported with its index."""
        result = await todo_resource.todo_write(todos=[TodoItem(content="Run tests", status="blocked", activeForm="Running tests")])

        assert result == "Error: Todo item 0 has invalid status 'blocked'"

    @pytest.mark.asyncio
    async def test_error_index_points_at_offending_item(self, todo_resource):
        """The reported index identifies the offending item in a mixed list."""
        result = await todo_resource.todo_write(
            todos=[
                TodoItem(content="First", status="pending", activeForm="Doing first"),
                {"content": "Second", "status": "nope", "activeForm": "Doing second"},
            ]
        )

        assert result == "Error: Todo item 1 has invalid status 'nope'"

    @pytest.mark.asyncio
    async def test_empty_list_renders_empty_progress(self, todo_resource):
        """An empty todo list is rendered rather than rejected."""
        result = await todo_resource.todo_write(todos=[])

        assert "Progress: 0/0 completed, 0 in progress, 0 pending" in result
