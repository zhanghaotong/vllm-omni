from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.tracing.utils import contains_trace_headers, extract_trace_headers, log_tracing_disabled_warning


async def get_trace_headers(
    engine_client: Any,
    headers: Mapping[str, str],
) -> Mapping[str, str] | None:
    """Return propagated trace headers when tracing is enabled for the engine."""
    is_tracing_enabled_fn = getattr(engine_client, "is_tracing_enabled", None)
    is_tracing_enabled = False

    if callable(is_tracing_enabled_fn):
        is_tracing_enabled = await is_tracing_enabled_fn()

    if is_tracing_enabled:
        return extract_trace_headers(headers)

    if contains_trace_headers(headers):
        log_tracing_disabled_warning()

    return None
