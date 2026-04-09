from __future__ import annotations

from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
    is_tracing_available,
)

logger = init_logger(__name__)

REQUEST_TRACE_SPAN_NAME = "omni.request"


def _seconds_to_ns(timestamp_s: float) -> int:
    return int(max(0.0, float(timestamp_s)) * 1_000_000_000)


def emit_request_trace(
    *,
    request_id: str,
    model_name: str,
    final_output_type: str,
    trace_headers: Mapping[str, str] | None,
    start_time_s: float,
    end_time_s: float,
    status: str,
    e2e_total_ms: float,
    e2e_total_tokens: int,
    transfers_total_time_ms: float,
    transfers_total_bytes: int,
) -> None:
    if trace_headers is None or not is_tracing_available():
        return

    attrs: dict[str, object] = {
        SpanAttributes.GEN_AI_REQUEST_ID: request_id,
        SpanAttributes.GEN_AI_RESPONSE_MODEL: model_name,
        "vllm.omni.final_output_type": final_output_type,
        "vllm.omni.request.status": status,
        "vllm.omni.request.e2e_ms": float(e2e_total_ms),
        "vllm.omni.request.total_tokens": int(e2e_total_tokens),
        "vllm.omni.transfer.total_time_ms": float(transfers_total_time_ms),
        "vllm.omni.transfer.total_bytes": int(transfers_total_bytes),
    }
    ctx = extract_trace_context(trace_headers)

    try:
        instrument_manual(
            span_name=REQUEST_TRACE_SPAN_NAME,
            start_time=_seconds_to_ns(start_time_s),
            end_time=_seconds_to_ns(end_time_s),
            attributes=attrs,
            context=ctx,
            kind=SpanKind.SERVER,
        )
    except Exception:
        logger.debug("Failed to emit request trace for %s", request_id, exc_info=True)
