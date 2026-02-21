"""Periodic health checker that benchmarks models and tracks availability"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

import litellm

from .config import Config
from .litellm_utils import build_litellm_params
from .models import HealthCheckResult

logger = logging.getLogger(__name__)

BENCH_PROMPT = [{"role": "user", "content": "[lazyrouter-health-check] Say hi"}]
BENCH_MAX_TOKENS = 16
BENCH_TEMPERATURE = 0.0
HEALTH_CHECK_HEADER = {"X-LazyRouter-Request-Type": "health-check"}


def is_result_healthy(result: HealthCheckResult, max_latency_ms: int) -> bool:
    """Return True when a health result is safe to use for routing."""
    return (
        result.status == "ok"
        and result.total_ms is not None
        and result.total_ms <= max_latency_ms
    )


def _compact_error(error: Exception, limit: int = 240) -> str:
    """Return a one-line, bounded error summary for logs."""
    error_str = str(error).strip()
    text = error_str.splitlines()[0] if error_str else repr(error)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _parse_stream_chunk_payload(chunk: Any) -> Optional[Dict[str, Any]]:
    """Extract JSON payload from an SSE chunk."""
    if isinstance(chunk, dict):
        return chunk

    if not isinstance(chunk, str):
        return None

    text = chunk.strip()
    if not text.startswith("data:"):
        return None

    payload_text = text[len("data:") :].strip()
    if not payload_text or payload_text == "[DONE]":
        return None

    try:
        parsed = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _chunk_has_text_delta(payload: Dict[str, Any]) -> bool:
    """Return True when a stream chunk carries textual assistant delta."""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content.strip():
            return True

    return False


class LiteLLMWrapper:
    """Wrapper to provide provider-like interface for LiteLLM"""

    def __init__(
        self, api_key: str, base_url: Optional[str], api_style: str, model: str
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.api_style = api_style
        self.model = model

    def _get_litellm_params(self, model: str) -> dict:
        """Build LiteLLM parameters"""
        return build_litellm_params(self.api_key, self.base_url, self.api_style, model)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Send chat completion via LiteLLM"""
        params = self._get_litellm_params(model)
        existing_headers = params.get("extra_headers")
        if isinstance(existing_headers, dict):
            params["extra_headers"] = {**existing_headers, **HEALTH_CHECK_HEADER}
        else:
            params["extra_headers"] = dict(HEALTH_CHECK_HEADER)
        params.update(
            {
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
            }
        )

        if max_tokens:
            params["max_tokens"] = max_tokens

        params.update(kwargs)

        try:
            response = await litellm.acompletion(**params)
        except Exception as e:
            logger.warning(
                "Health-check LiteLLM call failed for model=%s: %s", model, e
            )
            raise

        if stream:
            return self._wrap_stream(response)
        else:
            return response.model_dump(exclude_none=True)

    async def _wrap_stream(self, response) -> AsyncGenerator[str, None]:
        """Convert LiteLLM stream to SSE format"""
        async for chunk in response:
            chunk_dict = chunk.model_dump(exclude_none=True)
            yield f"data: {json.dumps(chunk_dict)}\n\n"
        yield "data: [DONE]\n\n"


async def check_model_health(
    name: str, provider, actual_model: str, provider_name: str, is_router: bool = False
) -> HealthCheckResult:
    """Check health using a streaming probe, with non-stream fallback."""
    ttft_ms = None
    ttft_source = None
    ttft_unavailable_reason = None
    total_ms = None
    first_event_ms = None
    try:
        t0 = time.monotonic()
        stream = await provider.chat_completion(
            model=actual_model,
            messages=BENCH_PROMPT,
            stream=True,
            temperature=BENCH_TEMPERATURE,
            max_tokens=BENCH_MAX_TOKENS,
        )

        async for chunk in stream:
            chunk_ms = round((time.monotonic() - t0) * 1000, 1)
            payload = _parse_stream_chunk_payload(chunk)
            if payload is None:
                continue
            if first_event_ms is None:
                first_event_ms = chunk_ms
            if ttft_ms is None and _chunk_has_text_delta(payload):
                ttft_ms = chunk_ms
                ttft_source = "stream_text"

        # Fallback when provider streams events without textual delta content.
        if ttft_ms is None:
            ttft_ms = first_event_ms
            if ttft_ms is not None:
                ttft_source = "stream_event"

        total_ms = round((time.monotonic() - t0) * 1000, 1)

        return HealthCheckResult(
            model=name,
            provider=provider_name,
            actual_model=actual_model,
            is_router=is_router,
            status="ok",
            is_healthy=None,  # Set later in run_check based on latency threshold
            ttft_ms=ttft_ms,
            ttft_source=ttft_source,
            ttft_unavailable_reason=ttft_unavailable_reason,
            total_ms=total_ms,
        )
    except Exception as stream_error:
        # Some providers/parsers can fail in stream mode even when non-streaming
        # completions are healthy. Fall back to non-stream probe to avoid false
        # unhealthy status caused only by TTFT measurement path.
        logger.debug(
            "Health-check stream probe failed for model=%s (%s); retrying non-stream: %s",
            name,
            actual_model,
            _compact_error(stream_error),
        )
        # Force TTFT fields to match the non-stream fallback source.
        ttft_ms = None
        ttft_source = "unavailable_non_stream"
        ttft_unavailable_reason = _compact_error(stream_error)
        try:
            t0 = time.monotonic()
            await provider.chat_completion(
                model=actual_model,
                messages=BENCH_PROMPT,
                stream=False,
                temperature=BENCH_TEMPERATURE,
                max_tokens=BENCH_MAX_TOKENS,
            )
            total_ms = round((time.monotonic() - t0) * 1000, 1)
            return HealthCheckResult(
                model=name,
                provider=provider_name,
                actual_model=actual_model,
                is_router=is_router,
                status="ok",
                is_healthy=None,  # Set later in run_check based on latency threshold
                ttft_ms=None,
                ttft_source=ttft_source,
                ttft_unavailable_reason=ttft_unavailable_reason,
                total_ms=total_ms,
            )
        except Exception as fallback_error:
            return HealthCheckResult(
                model=name,
                provider=provider_name,
                actual_model=actual_model,
                is_router=is_router,
                status="error",
                is_healthy=False,
                ttft_ms=ttft_ms,
                ttft_source=ttft_source,
                ttft_unavailable_reason=ttft_unavailable_reason,
                total_ms=total_ms,
                error=(
                    f"stream probe failed: {_compact_error(stream_error)}; "
                    f"non-stream probe failed: {_compact_error(fallback_error)}"
                ),
            )


class HealthChecker:
    """Runs periodic health checks and tracks which models are available."""

    def __init__(self, config: Config):
        self.config = config
        self.hc_config = config.health_check
        self.healthy_models: Set[str] = set(
            config.llms.keys()
        )  # assume all healthy at start
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.last_router_result: Optional[HealthCheckResult] = None
        self.last_check: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._run_lock = asyncio.Lock()
        self._activity_lock = asyncio.Lock()
        self._activity_event = asyncio.Event()
        self._last_request_at = time.monotonic()
        self._idle_mode_active = False

    @property
    def unhealthy_models(self) -> Set[str]:
        """Return set of models that failed the last health check."""
        return set(self.config.llms.keys()) - self.healthy_models

    def _seconds_since_last_request(self) -> float:
        """Return elapsed seconds since the last chat completion request."""
        return max(0.0, time.monotonic() - self._last_request_at)

    def _is_idle(self) -> bool:
        """Return True when chat traffic has been inactive long enough."""
        return self._seconds_since_last_request() >= self.hc_config.idle_after_seconds

    async def note_request_and_maybe_run_cold_boot_check(self) -> bool:
        """Mark request activity and run a health check when resuming from idle."""
        should_check = False
        idle_for = 0.0

        async with self._activity_lock:
            idle_for = self._seconds_since_last_request()
            should_check = self._is_idle()
            self._last_request_at = time.monotonic()
            self._activity_event.set()

        if not should_check:
            return False

        logger.info(
            "[health-check] pre-route check after %.1fs idle before serving request",
            idle_for,
        )
        try:
            await self.run_check()
        except Exception as e:
            logger.warning(
                "[health-check] pre-route check failed; continuing request with cached status: %s",
                e,
            )
        return True

    async def run_check(self) -> list[HealthCheckResult]:
        """Run a single health check against all configured models."""
        async with self._run_lock:
            return await self._run_check_once()

    async def _run_check_once(self) -> list[HealthCheckResult]:
        """Run a single health check against all configured models."""
        tasks = []
        model_names = []
        router_provider_name = self.config.router.provider
        router_model = self.config.router.model
        router_api_key = self.config.get_api_key(router_provider_name)
        router_base_url = self.config.get_base_url(router_provider_name)
        router_api_style = self.config.get_api_style(router_provider_name)
        router_provider = LiteLLMWrapper(
            router_api_key, router_base_url, router_api_style, router_model
        )
        router_task = asyncio.wait_for(
            check_model_health(
                router_model,
                router_provider,
                router_model,
                router_provider_name,
                is_router=True,
            ),
            timeout=self.hc_config.max_latency_ms / 1000 + 5,  # generous timeout
        )

        for model_name, model_config in self.config.llms.items():
            api_key = self.config.get_api_key(model_config.provider)
            base_url = self.config.get_base_url(model_config.provider)
            api_style = self.config.get_api_style(model_config.provider)

            # Create LiteLLM wrapper
            provider = LiteLLMWrapper(api_key, base_url, api_style, model_config.model)

            model_names.append(model_name)
            tasks.append(
                asyncio.wait_for(
                    check_model_health(
                        model_name, provider, model_config.model, model_config.provider
                    ),
                    timeout=self.hc_config.max_latency_ms / 1000
                    + 5,  # generous timeout
                )
            )

        gathered = await asyncio.gather(*tasks, router_task, return_exceptions=True)
        raw_results = gathered[:-1]
        raw_router_result = gathered[-1]

        results = []
        new_healthy = set()
        for i, r in enumerate(raw_results):
            name = model_names[i]
            if isinstance(r, HealthCheckResult):
                # Determine if model is healthy (available for routing)
                is_healthy = is_result_healthy(r, self.hc_config.max_latency_ms)
                r.is_healthy = is_healthy

                results.append(r)
                self.last_results[name] = r

                if is_healthy:
                    new_healthy.add(name)
                else:
                    reason = (
                        r.error
                        if r.status == "error"
                        else (
                            "total_ms unavailable"
                            if r.total_ms is None
                            else f"total_ms={r.total_ms} > {self.hc_config.max_latency_ms}"
                        )
                    )
                    logger.warning(f"Health check: {name} unhealthy — {reason}")
            else:
                err = "Timed out" if isinstance(r, asyncio.TimeoutError) else str(r)
                mc = self.config.llms[name]
                result = HealthCheckResult(
                    model=name,
                    provider=mc.provider,
                    actual_model=mc.model,
                    is_router=False,
                    status="error",
                    is_healthy=False,
                    error=err,
                )
                results.append(result)
                self.last_results[name] = result
                logger.warning(f"Health check: {name} unhealthy — {err}")

        if isinstance(raw_router_result, HealthCheckResult):
            router_result = raw_router_result
            router_result.is_healthy = is_result_healthy(
                router_result, self.hc_config.max_latency_ms
            )
            self.last_router_result = router_result
            if not router_result.is_healthy:
                reason = (
                    router_result.error
                    if router_result.status == "error"
                    else (
                        "total_ms unavailable"
                        if router_result.total_ms is None
                        else f"total_ms={router_result.total_ms} > {self.hc_config.max_latency_ms}"
                    )
                )
                logger.warning(f"Health check: router model unhealthy - {reason}")
        else:
            err = (
                "Timed out"
                if isinstance(raw_router_result, asyncio.TimeoutError)
                else str(raw_router_result)
            )
            self.last_router_result = HealthCheckResult(
                model=router_model,
                provider=router_provider_name,
                actual_model=router_model,
                is_router=True,
                status="error",
                is_healthy=False,
                error=err,
            )
            logger.warning(f"Health check: router model unhealthy - {err}")

        # If ALL models are unhealthy, log error and keep none available (strict mode)
        if not new_healthy:
            logger.error("Health check: ALL models unhealthy; keeping none available")

        if new_healthy != self.healthy_models:
            added = new_healthy - self.healthy_models
            removed = self.healthy_models - new_healthy
            if added:
                logger.info(f"Health check: models recovered: {added}")
            if removed:
                logger.info(f"Health check: models excluded: {removed}")

        self.healthy_models = new_healthy
        self.last_check = datetime.now(timezone.utc).isoformat()
        router_health = (
            "unknown"
            if self.last_router_result is None
            else ("healthy" if self.last_router_result.is_healthy else "unhealthy")
        )
        logger.info(
            f"Health check complete: {len(new_healthy)}/{len(self.config.llms)} models healthy; router={router_health}"
        )
        return results

    async def _loop(self):
        """Background loop that runs health checks periodically."""
        # Initial check on startup
        try:
            await self.run_check()
        except Exception as e:
            logger.error(f"Initial health check failed: {e}")

        while True:
            if self._is_idle():
                if not self._idle_mode_active:
                    self._idle_mode_active = True
                    logger.info(
                        "[health-check] idle mode enabled: pausing background checks after %.0fs inactivity",
                        self.hc_config.idle_after_seconds,
                    )
                await self._activity_event.wait()
                self._activity_event.clear()
                continue

            if self._idle_mode_active:
                self._idle_mode_active = False
                logger.info(
                    "[health-check] idle mode disabled: resuming %ss interval",
                    self.hc_config.interval,
                )

            await asyncio.sleep(self.hc_config.interval)
            try:
                await self.run_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    def start(self):
        """Start the background health check loop."""
        self._last_request_at = time.monotonic()
        self._activity_event.clear()
        logger.info(
            f"Starting health checks every {self.hc_config.interval}s "
            f"(max_latency={self.hc_config.max_latency_ms}ms)"
        )
        logger.info(
            "Background checks pause after %ss idle; first request after idle triggers pre-route health check",
            self.hc_config.idle_after_seconds,
        )
        self._task = asyncio.create_task(self._loop())

    def stop(self):
        """Stop the background health check loop."""
        if self._task:
            self._task.cancel()
            self._task = None

