from pathlib import Path

import pytest
import pytest_asyncio
import redis
import redis.asyncio as async_redis
from redis.exceptions import RedisError

from services.budget_redis import (
    counter_key as _counter_key,
)
from services.budget_redis import (
    marker_key as _marker_key,
)

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
    return _counter_key


@pytest.fixture
def marker_key():
    return _marker_key


@pytest_asyncio.fixture
async def async_redis_client():
    """Use isolated async Redis database 15 for wrapper tests."""
    client = async_redis.Redis(
        host="localhost",
        port=6379,
        db=15,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
    )

    try:
        await client.ping()
    except (RedisError, OSError) as exc:
        await client.aclose()
        pytest.skip(f"Redis is unavailable: {exc}")

    await client.flushdb()

    try:
        yield client
    finally:
        try:
            await client.flushdb()
        finally:
            await client.aclose()