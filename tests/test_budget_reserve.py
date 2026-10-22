import pytest
import redis
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Return codes from budget_reserve.lua
ACCEPTED = 1
ALREADY_RESERVED = 2
RECOVERY_REQUIRED = 3
CORRUPT = 4
OVERSIZED = 5
EXCEEDED = 6
NO_TTL = 7


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "budget_reserve.lua"

@pytest.fixture
def redis_client():
    client = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)

    # Verify connection during setup before running test or flushing DB
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis server is not available at localhost:6379")

    client.flushdb()

    yield client

    # Cleanup teardown
    try:
        client.flushdb()
    except redis.ConnectionError:
        pass

@pytest.fixture
def reserve_script(redis_client):
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        script = f.read()
   
    return redis_client.register_script(script)


def marker_key(period_id, reservation_id):
    return f"budget:marker:{period_id}:{reservation_id}"


# ============================================================
# 1. Accepted
# ============================================================

def test_reserve_accepted(redis_client, reserve_script):
    period_id = "202610"
    reservation_id = "reservation-1"

    counter_key = f"budget:committed:{period_id}"
    marker = marker_key(period_id, reservation_id)

    redis_client.set(counter_key, 100, ex=3600)

    result = reserve_script(
        keys=[counter_key, marker],
        args=[50, 1000],
    )

    assert result[0] == ACCEPTED
    assert result[1] == "accepted"
    assert int(result[2]) == 150
    assert int(result[3]) == 1000
    assert int(result[4]) == 850

    assert redis_client.get(counter_key) == "150"
    assert redis_client.get(marker) == "50"

    marker_ttl = redis_client.ttl(marker)
    counter_ttl = redis_client.ttl(counter_key)
    assert 0 < marker_ttl <= counter_ttl


# ============================================================
# 2. Already reserved / idempotent retry
# ============================================================

def test_reserve_already_reserved(redis_client, reserve_script):
    period_id = "202610"
    reservation_id = "reservation-2"

    counter_key = f"budget:committed:{period_id}"
    marker = marker_key(period_id, reservation_id)

    redis_client.set(counter_key, 100, ex=3600)
    redis_client.set(marker, 50, ex=3600)

    result = reserve_script(
        keys=[counter_key, marker],
        args=[50, 1000],
    )

    assert result[0] == ALREADY_RESERVED
    assert result[1] == "already_reserved"

    # Nothing changed.
    assert redis_client.get(counter_key) == "100"
    assert redis_client.get(marker) == "50"


# ============================================================
# 3. Missing counter -> recovery required
# ============================================================

def test_reserve_requires_recovery_when_counter_missing(
    redis_client,
    reserve_script,
):
    period_id = "202610"
    reservation_id = "reservation-3"

    counter_key = f"budget:committed:{period_id}"
    marker = marker_key(period_id, reservation_id)

    result = reserve_script(
        keys=[counter_key, marker],
        args=[50, 1000],
    )

    assert result[0] == RECOVERY_REQUIRED
    assert result[1] == "recovery_required"

    # Script must not create anything.
    assert redis_client.exists(counter_key) == 0
    assert redis_client.exists(marker) == 0


# ============================================================
# 4. Corrupt counter
# ============================================================

def test_reserve_rejects_corrupt_counter(
    redis_client,
    reserve_script,
):
    period_id = "202610"
    reservation_id = "reservation-4"

    counter_key = f"budget:committed:{period_id}"
    marker = marker_key(period_id, reservation_id)

    redis_client.set(counter_key, "not-an-integer", ex=3600)

    result = reserve_script(
        keys=[counter_key, marker],
        args=[50, 1000],
    )

    assert result[0] == CORRUPT
    assert result[1] == "corrupt"

    # Nothing changed.
    assert redis_client.get(counter_key) == "not-an-integer"
    assert redis_client.exists(marker) == 0


# ============================================================
# 5. Oversized reservation
# ============================================================

def test_reserve_rejects_counter_without_ttl(redis_client, reserve_script):
    counter_key = "budget:committed:202610"
    marker = marker_key("202610", "reservation-no-ttl")
    redis_client.set(counter_key, 100)

    result = reserve_script(
        keys=[counter_key, marker],
        args=[50, 1000],
    )

    assert result[0] == NO_TTL
    assert result[1] == "no_ttl"
    assert redis_client.get(counter_key) == "100"
    assert redis_client.exists(marker) == 0


def test_reserve_exact_fit_is_accepted(redis_client, reserve_script):
    counter_key = "budget:committed:exact-fit"
    marker = marker_key("exact-fit", "reservation")
    redis_client.set(counter_key, 900, ex=3600)

    result = reserve_script(keys=[counter_key, marker], args=[100, 1000])

    assert result[0] == ACCEPTED
    assert redis_client.get(counter_key) == "1000"


def test_reserve_at_limit_is_exceeded(redis_client, reserve_script):
    counter_key = "budget:committed:at-limit"
    marker = marker_key("at-limit", "reservation")
    redis_client.set(counter_key, 1000, ex=3600)

    result = reserve_script(keys=[counter_key, marker], args=[1, 1000])

    assert result[0] == EXCEEDED
    assert result[1] == "exceeded"
    assert redis_client.get(counter_key) == "1000"
    assert redis_client.exists(marker) == 0


def test_reservation_equal_to_limit_is_accepted(redis_client, reserve_script):
    counter_key = "budget:committed:limit-sized"
    marker = marker_key("limit-sized", "reservation")
    redis_client.set(counter_key, 0, ex=3600)

    result = reserve_script(keys=[counter_key, marker], args=[1000, 1000])

    assert result[0] == ACCEPTED
    assert redis_client.get(counter_key) == "1000"


@pytest.mark.parametrize("amount", [0, -1, "not-a-number"])
def test_reserve_rejects_invalid_amount_without_writes(
    redis_client, reserve_script, amount
):
    counter_key = "budget:committed:invalid-amount"
    marker = marker_key("invalid-amount", f"reservation-{amount}")
    redis_client.set(counter_key, 25, ex=3600)

    result = reserve_script(keys=[counter_key, marker], args=[amount, 1000])

    assert result[0] == -1
    assert result[1] == "invalid_arguments"
    assert redis_client.get(counter_key) == "25"
    assert redis_client.exists(marker) == 0


def test_concurrent_reservations_are_atomic(redis_client, reserve_script):
    counter_key = "budget:committed:concurrent"
    redis_client.set(counter_key, 0, ex=3600)

    def reserve(index):
        marker = marker_key("concurrent", f"reservation-{index}")
        return reserve_script(keys=[counter_key, marker], args=[100, 1000])

    with ThreadPoolExecutor(max_workers=50) as pool:
        results = list(pool.map(reserve, range(50)))

    assert sum(result[0] == ACCEPTED for result in results) == 10
    assert redis_client.get(counter_key) == "1000"
    markers = list(redis_client.scan_iter(match="budget:marker:concurrent:*"))
    assert len(markers) == 10


# ============================================================
# 6. Oversized reservation
# ============================================================

def test_reserve_rejects_oversized_reservation(
    redis_client,
    reserve_script,
):
    period_id = "202610"
    reservation_id = "reservation-5"

    counter_key = f"budget:committed:{period_id}"
    marker = marker_key(period_id, reservation_id)

    redis_client.set(counter_key, 100, ex=3600)

    result = reserve_script(
        keys=[counter_key, marker],
        args=[1500, 1000],
    )

    assert result[0] == OVERSIZED
    assert result[1] == "oversized"

    assert int(result[2]) == 100
    assert int(result[3]) == 1000
    assert int(result[4]) == 900

    # Nothing changed.
    assert redis_client.get(counter_key) == "100"
    assert redis_client.exists(marker) == 0


# ============================================================
# 7. Budget exceeded
# ============================================================

def test_reserve_rejects_when_budget_would_be_exceeded(
    redis_client,
    reserve_script,
):
    period_id = "202610"
    reservation_id = "reservation-6"

    counter_key = f"budget:committed:{period_id}"
    marker = marker_key(period_id, reservation_id)

    redis_client.set(counter_key, 900, ex=3600)

    result = reserve_script(
        keys=[counter_key, marker],
        args=[200, 1000],
    )

    assert result[0] == EXCEEDED
    assert result[1] == "exceeded"

    assert int(result[2]) == 900
    assert int(result[3]) == 1000
    assert int(result[4]) == 100

    # Nothing changed.
    assert redis_client.get(counter_key) == "900"
    assert redis_client.exists(marker) == 0
