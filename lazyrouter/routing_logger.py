"""Routing decision logger for analyzing router performance"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RoutingLogger:
    """Logs routing decisions to a separate JSONL file for easy analysis"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"routing_{timestamp}.jsonl"
        logger.info(f"Routing decisions will be logged to {self.log_path}")

    def log_routing_decision(
        self,
        *,
        request_id: str,
        context: str,
        model_descriptions: str,
        selected_model: str,
        router_response: Optional[str],
        context_length: int,
        num_context_messages: int,
        latency_ms: float,
    ) -> None:
        """Log a routing decision

        Args:
            request_id: Unique request identifier
            context: The context sent to the router
            model_descriptions: Available model descriptions
            selected_model: The model selected by the router
            router_response: Raw response from router (JSON)
            context_length: Length of context in characters
            num_context_messages: Number of messages included in context
            latency_ms: Router latency in milliseconds
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "selected_model": selected_model,
            "context_length": context_length,
            "num_context_messages": num_context_messages,
            "latency_ms": round(latency_ms, 2),
            "context": context,
            "model_descriptions": model_descriptions,
            "router_response": router_response,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
