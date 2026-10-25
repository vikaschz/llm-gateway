import asyncio
import time
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base
from services import budget_redis as budget
from services.budget_reservation import BudgetReservation


@pytest_asyncio.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "budget-redis.db"

    sync_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    try:
        yield async_sessionmaker(
            engine,
            expire_on_commit=False,
        )
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(session_factory, monkeypatch):
    """Provides an active DB session and mocks the rebuild snapshot."""
    async with session_factory() as session:
        monkeypatch.setattr(
            budget,
            "get_rebuild_snapshot",
            AsyncMock(
                return_value=(
                    400,
                    [
                        ("reservation-1", 100),
                        ("reservation-2", 250),
                        ("reservation-3", 50),
                    ],
                )
            ),
        )
        yield session


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (1, budget.ReserveOutcome.ACCEPTED),
        (2, budget.ReserveOutcome.ALREADY_RESERVED),
        (3, budget.ReserveOutcome.RECOVERY_REQUIRED),
        (4, budget.ReserveOutcome.CORRUPT),
        (5, budget.ReserveOutcome.OVERSIZED),
        (6, budget.ReserveOutcome.EXCEEDED),
        (7, budget.ReserveOutcome.NO_TTL),
        (-1, budget.ReserveOutcome.INVALID_ARGUMENTS),
    ],
)
def test_reserve_return_codes_map_to_named_outcomes(code, expected):
    result = budget._reserve_result([code, "message", 12, 100, 88])

    assert result.outcome is expected
    assert result.committed == 12
    assert result.limit == 100
    assert result.remaining == 88

    assert all(
        type(n) is int
        for n in (
            result.committed,
            result.limit,
            result.remaining,
        )
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (1, budget.SettleOutcome.SETTLED),
        (2, budget.SettleOutcome.MARKER_MISSING),
        (3, budget.SettleOutcome.COUNTER_MISSING),
        (4, budget.SettleOutcome.NEGATIVE),
        (5, budget.SettleOutcome.CORRUPT),
        (-1, budget.SettleOutcome.INVALID_ARGUMENTS),
    ],
)
def test_settle_return_codes_map_to_named_outcomes(code, expected):
    result = budget._settle_result([code, "message", 25, -5])

    assert result.outcome is expected
    assert result.committed == 25
    assert result.delta == -5

    assert type(result.committed) is int
    assert type(result.delta) is int


def test_unknown_reserve_code_raises():
    with pytest.raises(
        ValueError,
        match="Unknown budget reserve return code",
    ):
        budget._reserve_result([999, "unknown"])


def test_unknown_settle_code_raises():
    with pytest.raises(
        ValueError,
        match="Unknown budget settle return code",
    ):
        budget._settle_result([999, "unknown"])


def test_key_helpers():
    assert budget.counter_key("202610") == "budget:committed:202610"

    assert budget.marker_key("202610", "abc") == "budget:marker:202610:abc"


@pytest.mark.asyncio
async def test_reserve_and_settle_with_redis(async_redis_client):
    period = "202610"
    reservation_id = "wrapper-round-trip"

    counter = budget.counter_key(period)
    marker = budget.marker_key(
        period,
        reservation_id,
    )

    await async_redis_client.set(
        counter,
        100,
        ex=3600,
    )

    reserved = await budget.reserve_budget(
        async_redis_client,
        period,
        reservation_id,
        amount=25,
        limit=1000,
    )

    assert reserved.outcome is budget.ReserveOutcome.ACCEPTED

    assert reserved.committed == 125
    assert reserved.limit == 1000
    assert reserved.remaining == 875

    assert await async_redis_client.get(marker) == "25"

    settled = await budget.settle_budget(
        async_redis_client,
        period,
        reservation_id,
        actual_cost=10,
    )

    assert settled.outcome is budget.SettleOutcome.SETTLED

    assert settled.committed == 110
    assert settled.delta == -15

    assert await async_redis_client.get(counter) == "110"

    assert await async_redis_client.exists(marker) == 0


# ============================================================
# ensure_counter()
# ============================================================


@pytest.mark.asyncio
async def test_ensure_counter_publishes_counter_and_markers(
    async_redis_client,
    db_session,
):
    period = "202610"
    expires_at = int(time.time()) + 3600

    outstanding = [
        ("reservation-1", 100),
        ("reservation-2", 250),
        ("reservation-3", 50),
    ]

    total = 400

    result = await budget.ensure_counter(
        async_redis_client,
        db_session,
        period,
        expires_at,
    )

    assert result is True

    counter = budget.counter_key(period)

    assert await async_redis_client.get(counter) == str(total)

    for reservation_id, amount in outstanding:
        marker = budget.marker_key(
            period,
            reservation_id,
        )

        assert await async_redis_client.get(marker) == str(amount)

        counter_expiry = await async_redis_client.execute_command(
            "PEXPIRETIME",
            counter,
        )

        marker_expiry = await async_redis_client.execute_command(
            "PEXPIRETIME",
            marker,
        )

        assert marker_expiry == counter_expiry


@pytest.mark.asyncio
async def test_ensure_counter_does_not_overwrite_existing_counter(
    async_redis_client,
    db_session,
):
    period = "202610"
    counter = budget.counter_key(period)

    expires_at = int(time.time()) + 3600
    await async_redis_client.set(
        counter,
        999,
        ex=3600,
    )

    result = await budget.ensure_counter(
        async_redis_client,
        db_session,
        period,
        expires_at,
    )

    assert result is False

    assert await async_redis_client.get(counter) == "999"


@pytest.mark.asyncio
async def test_ensure_counter_concurrent_publishers(
    async_redis_client,
    db_session,
):
    period = "202610"

    expires_at = int(time.time()) + 3600

    async def publish():
        return await budget.ensure_counter(
            async_redis_client,
            db_session,
            period,
            expires_at,
        )

    results = await asyncio.gather(*(publish() for _ in range(12)))

    # Exactly one process/thread should win the publication.
    assert sum(result is True for result in results) == 1

    counter = budget.counter_key(period)

    assert await async_redis_client.exists(counter) == 1

    # The DB snapshot determines the final counter.
    assert await async_redis_client.get(counter) == "400"

    expected_markers = {
        budget.marker_key(period, reservation_id)
        for reservation_id in (
            "reservation-1",
            "reservation-2",
            "reservation-3",
        )
    }

    actual_markers = {
        key for key in await async_redis_client.keys(f"budget:marker:{period}:*")
    }

    assert actual_markers == expected_markers


@pytest.mark.asyncio
async def test_ensure_counter_removes_lock_after_error(
    async_redis_client,
    db_session,
    monkeypatch,
):
    period = "202610"

    expires_at = int(time.time()) + 3600

    original_set = async_redis_client.set

    calls = 0

    async def failing_set(*args, **kwargs):
        nonlocal calls
        calls += 1

        # First SET creates the lock.
        # Fail when the function tries to publish the first marker.
        if calls == 2:
            raise RuntimeError("simulated publish failure")

        return await original_set(*args, **kwargs)

    monkeypatch.setattr(
        async_redis_client,
        "set",
        failing_set,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated publish failure",
    ):
        await budget.ensure_counter(
            async_redis_client,
            db_session,
            period,
            expires_at,
        )

    lock_key = f"budget:rebuild-lock:{period}"

    assert await async_redis_client.exists(lock_key) == 0


@pytest.mark.asyncio
async def test_ensure_counter_waits_for_stale_lock(
    async_redis_client,
    db_session,
):
    period = "202610"

    expires_at = int(time.time()) + 3600
    lock_key = f"budget:rebuild-lock:{period}"

    # Simulate another publisher that has become stale.
    await async_redis_client.set(
        lock_key,
        "stale-owner",
        px=100,
    )

    result = await budget.ensure_counter(
        async_redis_client,
        db_session,
        period,
        expires_at,
        wait_timeout=2.0,
    )

    assert result is True

    counter = budget.counter_key(period)

    assert await async_redis_client.get(counter) == "400"


@pytest.mark.asyncio
async def test_ensure_counter_times_out_on_permanent_lock(
    async_redis_client,
    db_session,
):
    period = "202610"

    expires_at = int(time.time()) + 3600
    lock_key = f"budget:rebuild-lock:{period}"

    # No expiry: this lock cannot recover automatically.
    await async_redis_client.set(
        lock_key,
        "permanent-owner",
    )

    with pytest.raises(TimeoutError):
        await budget.ensure_counter(
            async_redis_client,
            db_session,
            period,
            expires_at,
            wait_timeout=0.2,
        )

    counter = budget.counter_key(period)

    assert await async_redis_client.exists(counter) == 0

    # The waiter does not own the lock and must not delete it.
    assert await async_redis_client.get(lock_key) == "permanent-owner"


@pytest.mark.asyncio
async def test_ensure_counter_rebuilds_from_real_database(
    async_redis_client,
    session_factory,
):
    period_id = "202610"

    async with session_factory() as session:
        rows = [
            BudgetReservation(
                id="settled-current",
                request_id="request-settled-current",
                period_id=period_id,
                model="openai/gpt-oss-20b",
                state="settled",
                reserved_amount=100,
                actual_cost=35,
            ),
            BudgetReservation(
                id="reserved-current",
                request_id="request-reserved-current",
                period_id=period_id,
                model="openai/gpt-oss-20b",
                state="reserved",
                reserved_amount=80,
            ),
            BudgetReservation(
                id="uncertain-current",
                request_id="request-uncertain-current",
                period_id=period_id,
                model="openai/gpt-oss-20b",
                state="uncertain",
                reserved_amount=60,
            ),
            BudgetReservation(
                id="released-current",
                request_id="request-released-current",
                period_id=period_id,
                model="openai/gpt-oss-20b",
                state="released",
                reserved_amount=40,
            ),
            BudgetReservation(
                id="pending-current",
                request_id="request-pending-current",
                period_id=period_id,
                model="openai/gpt-oss-20b",
                state="pending",
                reserved_amount=50,
            ),
            BudgetReservation(
                id="settled-other-period",
                request_id="request-settled-other-period",
                period_id="202609",
                model="openai/gpt-oss-20b",
                state="settled",
                reserved_amount=200,
                actual_cost=200,
            ),
        ]

        session.add_all(rows)
        await session.commit()

        expires_at = int(time.time()) + 3600

        published = await budget.ensure_counter(
            async_redis_client,
            session,
            period_id,
            expires_at,
        )

    assert published is True

    assert await async_redis_client.get(budget.counter_key(period_id)) == "175"

    expected_markers = {
        budget.marker_key(period_id, "reserved-current"): "80",
        budget.marker_key(period_id, "uncertain-current"): "60",
    }

    for key, expected_value in expected_markers.items():
        assert await async_redis_client.get(key) == expected_value

    actual_markers = {
        key.decode() if isinstance(key, bytes) else key
        for key in await async_redis_client.keys(f"budget:marker:{period_id}:*")
    }

    assert actual_markers == set(expected_markers)


@pytest.mark.asyncio
async def test_ensure_counter_rebuilds_from_real_database_rows(
    async_redis_client,
    session_factory,
):
    period = "202610"
    expires_at = int(time.time()) + 3600

    def make_row(
        reservation_id,
        state,
        reserved_amount,
        actual_cost=None,
        period_id=period,
    ):
        return BudgetReservation(
            id=reservation_id,
            request_id=f"request-{reservation_id}",
            period_id=period_id,
            model="openai/gpt-oss-20b",
            state=state,
            reserved_amount=reserved_amount,
            actual_cost=actual_cost,
        )

    async with session_factory() as session:
        session.add_all(
            [
                make_row("settled-1", "settled", 100, actual_cost=35),
                make_row("reserved-1", "reserved", 80),
                make_row("uncertain-1", "uncertain", 60),
                make_row("released-1", "released", 40),
                make_row("pending-1", "pending", 50),
                # A different month must not be counted.
                make_row(
                    "other-month",
                    "settled",
                    500,
                    actual_cost=500,
                    period_id="202609",
                ),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        published = await budget.ensure_counter(
            async_redis_client,
            session,
            period,
            expires_at,
        )

    assert published is True

    # settled 35 + reserved 80 + uncertain 60; pending and released count zero.
    assert await async_redis_client.get(budget.counter_key(period)) == "175"

    markers = set(await async_redis_client.keys(f"budget:marker:{period}:*"))
    assert markers == {
        budget.marker_key(period, "reserved-1"),
        budget.marker_key(period, "uncertain-1"),
    }
    assert await async_redis_client.get(budget.marker_key(period, "reserved-1")) == "80"
    assert await async_redis_client.get(budget.marker_key(period, "uncertain-1")) == "60"

    # The other month was never touched.
    assert await async_redis_client.exists(budget.counter_key("202609")) == 0


@pytest.mark.asyncio
async def test_ensure_counter_initializes_zero_for_a_new_month(
    async_redis_client,
    session_factory,
):
    period = "202611"
    expires_at = int(time.time()) + 3600

    async with session_factory() as session:
        published = await budget.ensure_counter(
            async_redis_client,
            session,
            period,
            expires_at,
        )

    assert published is True
    assert await async_redis_client.get(budget.counter_key(period)) == "0"
    assert await async_redis_client.keys(f"budget:marker:{period}:*") == []