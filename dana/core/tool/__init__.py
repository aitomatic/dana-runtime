"""
Tool package — ToolExecutor, helpers, schema generation, Tool Catalog,
and Tool Execution Engine (D2).

Extracted from dana.core.runtime and dana.core.agent.components.
"""

from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity
from dana.core.tool.execution_engine import ToolExecutionEngine
from dana.core.tool.identity import check_collision
from dana.core.tool.tool_executor import ToolExecutor
from dana.core.tool.tool_executor_helpers import ToolExecutorHelpers
from dana.core.tool.tool_schema import generate_tool_schemas


__all__ = [
    "ToolCatalog",
    "ToolCatalogEntry",
    "ToolExecutionEngine",
    "ToolExecutor",
    "ToolExecutorHelpers",
    "ToolIdentity",
    "check_collision",
    "generate_tool_schemas",
]
