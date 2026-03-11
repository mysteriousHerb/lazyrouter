import pytest

from lazyrouter.config import HealthCheckConfig


def test_health_check_idle_after_seconds_must_be_positive():
    with pytest.raises(ValueError, match="health_check.idle_after_seconds must be > 0"):
        HealthCheckConfig(idle_after_seconds=0)
