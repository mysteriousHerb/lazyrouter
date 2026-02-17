"""Tests for retry handler functionality"""


from lazyrouter.config import ModelConfig
from lazyrouter.retry_handler import (
    get_model_elo,
    is_retryable_error,
    select_fallback_models,
)


class TestIsRetryableError:
    """Tests for is_retryable_error function"""

    def test_rate_limit_429_is_retryable(self):
        assert is_retryable_error(Exception("Error 429: Rate limit exceeded"))

    def test_rate_limit_text_is_retryable(self):
        assert is_retryable_error(Exception("rate limit reached"))

    def test_service_unavailable_503_is_retryable(self):
        assert is_retryable_error(Exception("503 Service Unavailable"))

    def test_bad_gateway_502_is_retryable(self):
        assert is_retryable_error(Exception("502 Bad Gateway"))

    def test_gateway_timeout_504_is_retryable(self):
        assert is_retryable_error(Exception("504 Gateway Timeout"))

    def test_connection_reset_is_retryable(self):
        assert is_retryable_error(Exception("Connection reset by peer"))

    def test_connection_refused_is_retryable(self):
        assert is_retryable_error(Exception("Connection refused"))

    def test_timeout_is_retryable(self):
        assert is_retryable_error(Exception("Request timed out"))

    def test_overloaded_is_retryable(self):
        assert is_retryable_error(Exception("API is overloaded"))

    def test_bad_request_400_not_retryable(self):
        assert not is_retryable_error(Exception("400 Bad Request"))

    def test_unauthorized_401_not_retryable(self):
        assert not is_retryable_error(Exception("401 Unauthorized"))

    def test_not_found_404_not_retryable(self):
        assert not is_retryable_error(Exception("404 Not Found"))

    def test_validation_error_not_retryable(self):
        assert not is_retryable_error(Exception("Invalid parameter: temperature"))


class TestGetModelElo:
    """Tests for get_model_elo function"""

    def test_average_of_coding_and_writing(self):
        cfg = ModelConfig(
            provider="test",
            model="test",
            description="test",
            coding_elo=1400,
            writing_elo=1500,
        )
        assert get_model_elo(cfg) == 1450

    def test_coding_only(self):
        cfg = ModelConfig(
            provider="test",
            model="test",
            description="test",
            coding_elo=1400,
        )
        assert get_model_elo(cfg) == 1400

    def test_writing_only(self):
        cfg = ModelConfig(
            provider="test",
            model="test",
            description="test",
            writing_elo=1500,
        )
        assert get_model_elo(cfg) == 1500

    def test_no_elo(self):
        cfg = ModelConfig(
            provider="test",
            model="test",
            description="test",
        )
        assert get_model_elo(cfg) == 0


class TestSelectFallbackModels:
    """Tests for select_fallback_models function"""

    def _make_models(self):
        return {
            "high-elo": ModelConfig(
                provider="test", model="high", description="high", coding_elo=1500
            ),
            "mid-elo": ModelConfig(
                provider="test", model="mid", description="mid", coding_elo=1400
            ),
            "low-elo": ModelConfig(
                provider="test", model="low", description="low", coding_elo=1300
            ),
        }

    def test_selects_similar_elo_models(self):
        models = self._make_models()
        fallbacks = select_fallback_models("mid-elo", models)
        # Should prefer models closest to mid-elo (1400)
        # high-elo (1500) is 100 away, low-elo (1300) is 100 away
        assert len(fallbacks) == 2
        assert "mid-elo" not in fallbacks

    def test_excludes_already_tried(self):
        models = self._make_models()
        fallbacks = select_fallback_models(
            "mid-elo", models, already_tried={"high-elo"}
        )
        assert "high-elo" not in fallbacks
        assert "low-elo" in fallbacks

    def test_prefers_healthy_models(self):
        models = self._make_models()
        # Only low-elo is healthy
        fallbacks = select_fallback_models(
            "mid-elo", models, healthy_models={"low-elo"}
        )
        # low-elo should come first even though high-elo has closer ELO
        assert fallbacks[0] == "low-elo"

    def test_limits_to_max_fallback_models(self):
        # Create many models
        models = {
            f"model-{i}": ModelConfig(
                provider="test",
                model=f"m{i}",
                description=f"m{i}",
                coding_elo=1400 + i * 10,
            )
            for i in range(10)
        }
        fallbacks = select_fallback_models("model-5", models)
        # MAX_FALLBACK_MODELS is 3, so we get at most 2 fallbacks (3-1 for original)
        assert len(fallbacks) <= 2
