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

BENCH_PROMPT = [{"role": "user", "content": "Say hi"}]
BENCH_MAX_TOKENS = 16
BENCH_TEMPERATURE = 0.0


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
    """Check health of a single model with one non-streaming request."""
    ttft_ms = None
    total_ms = None
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
            ttft_ms=ttft_ms,
            total_ms=total_ms,
        )
    except Exception as e:
        return HealthCheckResult(
            model=name,
            provider=provider_name,
            actual_model=actual_model,
            is_router=is_router,
            status="error",
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            error=str(e),
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
        self.last_check: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def unhealthy_models(self) -> Set[str]:
        """Return set of models that failed the last health check."""
        return set(self.config.llms.keys()) - self.healthy_models

    async def run_check(self) -> list[HealthCheckResult]:
        """Run a single health check against all configured models."""
        tasks = []
        model_names = []

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

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        new_healthy = set()
        for i, r in enumerate(raw_results):
            name = model_names[i]
            if isinstance(r, HealthCheckResult):
                results.append(r)
                self.last_results[name] = r
                if (
                    r.status == "ok"
                    and r.total_ms is not None
                    and r.total_ms <= self.hc_config.max_latency_ms
                ):
                    new_healthy.add(name)
                else:
                    reason = (
                        r.error
                        if r.status == "error"
                        else f"total_ms={r.total_ms} > {self.hc_config.max_latency_ms}"
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
                    error=err,
                )
                results.append(result)
                self.last_results[name] = result
                logger.warning(f"Health check: {name} unhealthy — {err}")

        # Edge case: if ALL models are unhealthy, either fail-open or keep none.
        if not new_healthy:
            if getattr(self.hc_config, "fail_open_when_all_unhealthy", False):
                logger.warning(
                    "Health check: ALL models unhealthy, fail-open enabled; keeping all available"
                )
                new_healthy = set(self.config.llms.keys())
            else:
                logger.error(
                    "Health check: ALL models unhealthy, strict mode enabled; keeping none available"
                )

        if new_healthy != self.healthy_models:
            added = new_healthy - self.healthy_models
            removed = self.healthy_models - new_healthy
            if added:
                logger.info(f"Health check: models recovered: {added}")
            if removed:
                logger.info(f"Health check: models excluded: {removed}")

        self.healthy_models = new_healthy
        self.last_check = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"Health check complete: {len(new_healthy)}/{len(self.config.llms)} models healthy"
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
            await asyncio.sleep(self.hc_config.interval)
            try:
                await self.run_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    def start(self):
        """Start the background health check loop."""
        if not self.hc_config.enabled:
            logger.info("Health checks disabled")
            return
        logger.info(
            f"Starting health checks every {self.hc_config.interval}s "
            f"(max_latency={self.hc_config.max_latency_ms}ms)"
        )
        self._task = asyncio.create_task(self._loop())

    def stop(self):
        """Stop the background health check loop."""
        if self._task:
            self._task.cancel()
            self._task = None
