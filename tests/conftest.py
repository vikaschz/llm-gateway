from pathlib import Path

import pytest
import redis

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESERVE_SCRIPT_PATH = PROJECT_ROOT / "budget_reserve.lua"
SETTLE_SCRIPT_PATH = PROJECT_ROOT / "budget_settle.lua"


@pytest.fixture
def redis_client():
    """Use isolated Redis database 15 for tests."""
    client = redis.Redis(
        host="localhost",
        port=6379,
        db=15,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
    )

    try:
        client.ping()
    except (redis.exceptions.RedisError, OSError) as exc:
        client.close()
        pytest.skip(f"Redis is unavailable: {exc}")

    # Only flush the dedicated test database.
    client.flushdb()

    try:
        yield client
    finally:
        try:
            client.flushdb()
        finally:
            client.close()


@pytest.fixture
def reserve_script(redis_client):
    """Load budget_reserve.lua."""
    script = RESERVE_SCRIPT_PATH.read_text(encoding="utf-8")
    return redis_client.register_script(script)


@pytest.fixture
def settle_script(redis_client):
    """Load budget_settle.lua."""
    script = SETTLE_SCRIPT_PATH.read_text(encoding="utf-8")
    return redis_client.register_script(script)


@pytest.fixture
def counter_key():
    """Generate the monthly budget counter key."""
    return lambda period_id: f"budget:committed:{period_id}"


@pytest.fixture
def marker_key():
    """Generate a reservation marker key."""
    return lambda period_id, reservation_id: (
        f"budget:marker:{period_id}:{reservation_id}"
    )