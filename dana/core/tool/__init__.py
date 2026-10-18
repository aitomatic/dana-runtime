"""
Tool package — ToolExecutor, helpers, schema generation, Tool Catalog,
and Tool Execution Engine (D2).

Extracted from dana.core.runtime and dana.core.agent.components.
"""

from dana.core.tool.admission import AdmissionController, get_default_admission_controller
from dana.core.tool.catalog import ToolCatalog, ToolCatalogEntry, ToolIdentity
from dana.core.tool.effect_ordering import EffectProfile, plan_batch, plan_waves
from dana.core.tool.execution_engine import ToolExecutionEngine
from dana.core.tool.identity import check_collision
from dana.core.tool.native_catalog import build_native_tool_catalog, native_effect_metadata
from dana.core.tool.offload import (
    BlockingIOPool,
    BlockingIOShutdown,
    bind_blocking_io_pool,
    blocking_io,
    blocking_io_max_workers,
    bound_blocking_io_pool,
    current_blocking_io_pool,
    current_physical_work,
    drain_physical_work,
    get_default_blocking_io_pool,
    is_blocking_io,
    is_loop_affine,
    is_shutdown,
    loop_affine,
    run_blocking,
    run_blocking_sync,
    shutdown_default_blocking_io_executor,
    use_blocking_io_pool,
)
from dana.core.tool.tool_executor import ToolExecutor
from dana.core.tool.tool_executor_helpers import ToolExecutorHelpers
from dana.core.tool.tool_schema import generate_tool_schemas


__all__ = [
    "AdmissionController",
    "BlockingIOPool",
    "BlockingIOShutdown",
    "EffectProfile",
    "ToolCatalog",
    "ToolCatalogEntry",
    "ToolExecutionEngine",
    "ToolExecutor",
    "ToolExecutorHelpers",
    "ToolIdentity",
    "bind_blocking_io_pool",
    "blocking_io",
    "blocking_io_max_workers",
    "bound_blocking_io_pool",
    "build_native_tool_catalog",
    "check_collision",
    "current_blocking_io_pool",
    "current_physical_work",
    "drain_physical_work",
    "generate_tool_schemas",
    "get_default_admission_controller",
    "get_default_blocking_io_pool",
    "is_blocking_io",
    "is_loop_affine",
    "is_shutdown",
    "loop_affine",
    "native_effect_metadata",
    "plan_batch",
    "plan_waves",
    "run_blocking",
    "run_blocking_sync",
    "shutdown_default_blocking_io_executor",
    "use_blocking_io_pool",
]
