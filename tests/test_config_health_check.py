import pytest

from lazyrouter.config import HealthCheckConfig


def test_health_check_idle_after_seconds_must_be_positive():
    with pytest.raises(ValueError, match=r"health_check\.idle_after_seconds must be > 0"):
        HealthCheckConfig(idle_after_seconds=0)


def test_health_check_stagger_seconds_must_be_non_negative():
    with pytest.raises(ValueError, match=r"health_check\.stagger_seconds must be >= 0"):
        HealthCheckConfig(stagger_seconds=-1)
