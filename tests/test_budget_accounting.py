import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base
from services.budget_accounting import (
    calculate_rebuild_total,
    confirm_reservation,
    create_pending_reservation,
    get_rebuild_total,
    mark_reservation_uncertain,
    release_reservation,
    settle_reservation,
    transition_reservation,
)
from services.budget_reservation import BudgetReservation
from services.budget_transitions import LEGAL_TRANSITIONS


@pytest_asyncio.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "budget-accounting.db"
    sync_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def add_reservation(
    session,
    reservation_id,
    *,
    state="pending",
    period_id="202610",
    reserved_amount=100,
    actual_cost=None,
):
    if state == "settled" and actual_cost is None:
        actual_cost = 35
    row = BudgetReservation(
        id=reservation_id,
        request_id=f"request-{reservation_id}",
        period_id=period_id,
        model="test-model",
        state=state,
        reserved_amount=reserved_amount,
        actual_cost=actual_cost,
    )
    session.add(row)
    await session.commit()
    return row


async def read_row(session_factory, reservation_id):
    async with session_factory() as session:
        return await session.scalar(
            select(BudgetReservation).where(
                BudgetReservation.id == reservation_id
            )
        )


@pytest.mark.asyncio
async def test_settle_twice_keeps_first_cost(session_factory):
    async with session_factory() as session:
        await add_reservation(session, "settle-twice", state="reserved")
        first = await settle_reservation(session, "settle-twice", 35)
        await session.commit()
        second = await settle_reservation(session, "settle-twice", 99)
        await session.commit()

    assert first.changed is True
    assert second.changed is False
    assert second.reason == "already_in_target_state"
    row = await read_row(session_factory, "settle-twice")
    assert row.state == "settled"
    assert row.actual_cost == 35


@pytest.mark.asyncio
async def test_settling_released_row_conflicts_without_cost(session_factory):
    async with session_factory() as session:
        await add_reservation(session, "released", state="released")
        result = await settle_reservation(session, "released", 35)
        await session.commit()

    assert result.changed is False
    assert result.reason == "conflicting_state"
    row = await read_row(session_factory, "released")
    assert row.state == "released"
    assert row.actual_cost is None


@pytest.mark.asyncio
async def test_confirm_pending_reservation(session_factory):
    async with session_factory() as session:
        await add_reservation(session, "confirm")
        result = await confirm_reservation(session, "confirm")
        await session.commit()
        again = await confirm_reservation(session, "confirm")
        await session.commit()

    assert result.changed is True
    assert again.changed is False
    assert again.reason == "already_in_target_state"
    row = await read_row(session_factory, "confirm")
    assert row.state == "reserved"
    assert row.reserved_at is not None


@pytest.mark.asyncio
async def test_every_state_pair_matches_legal_transitions(session_factory):
    states = {state for pair in LEGAL_TRANSITIONS for state in pair}
    states.add("pending")

    for source in sorted(states):
        for target in sorted(states):
            reservation_id = f"pair-{source}-{target}"
            async with session_factory() as session:
                await add_reservation(session, reservation_id, state=source)
                if target == "reserved":
                    operation = confirm_reservation(session, reservation_id)
                elif target == "settled":
                    operation = settle_reservation(session, reservation_id, 35)
                elif target == "released":
                    operation = release_reservation(session, reservation_id, "test")
                elif target == "uncertain":
                    operation = mark_reservation_uncertain(session, reservation_id)
                else:
                    with pytest.raises(ValueError):
                        await transition_reservation(
                            session, reservation_id, "pending"
                        )
                    await session.commit()
                    row = await session.scalar(
                        select(BudgetReservation).where(
                            BudgetReservation.id == reservation_id
                        )
                    )
                    assert row.state == source
                    continue

                result = await operation
                await session.commit()

            assert result.changed == ((source, target) in LEGAL_TRANSITIONS)
            row = await read_row(session_factory, reservation_id)
            assert row.state == (target if result.changed else source)


@pytest.mark.asyncio
async def test_concurrent_settlement_changes_once(session_factory):
    async with session_factory() as session:
        await add_reservation(session, "race", state="reserved")

    async def settle(cost):
        async with session_factory() as session:
            result = await settle_reservation(session, "race", cost)
            await session.commit()
            return result

    results = await asyncio.gather(settle(35), settle(99))

    assert sum(result.changed for result in results) == 1
    row = await read_row(session_factory, "race")
    assert row.state == "settled"
    assert row.actual_cost in {35, 99}


@pytest.mark.asyncio
async def test_rebuild_total_counts_only_billable_states_in_period(session_factory):
    async with session_factory() as session:
        await add_reservation(session, "settled", state="settled", reserved_amount=100, actual_cost=35)
        await add_reservation(session, "reserved", state="reserved", reserved_amount=80)
        await add_reservation(session, "uncertain", state="uncertain", reserved_amount=60)
        await add_reservation(session, "released-rebuild", state="released", reserved_amount=40)
        await add_reservation(session, "pending-rebuild", state="pending", reserved_amount=50)
        await add_reservation(
            session, "other-period", state="settled", period_id="202611",
            reserved_amount=100, actual_cost=700,
        )
        total = await get_rebuild_total(session, "202610")
        period_rows = list((await session.scalars(
            select(BudgetReservation).where(BudgetReservation.period_id == "202610")
        )).all())

    assert total == 175
    assert total == calculate_rebuild_total(period_rows)
    rows = {row.id: row for row in period_rows}
    assert rows["settled"].actual_cost == 35
    assert rows["reserved"].reserved_amount == 80
    assert rows["uncertain"].reserved_amount == 60
    assert rows["released-rebuild"].state == "released"
    assert rows["pending-rebuild"].state == "pending"
    assert (await read_row(session_factory, "other-period")).actual_cost == 700


@pytest.mark.asyncio
async def test_create_pending_rejects_invalid_period_and_zero_amount(session_factory):
    invalid_inputs = [("bad-period-2026", "2026", 1),
                      ("bad-period-abcdef", "abcdef", 1),
                      ("zero-amount", "202610", 0)]
    async with session_factory() as session:
        for reservation_id, period_id, amount in invalid_inputs:
            with pytest.raises(ValueError):
                await create_pending_reservation(
                    session,
                    reservation_id=reservation_id,
                    request_id=f"request-{reservation_id}",
                    period_id=period_id,
                    model="test-model",
                    reserved_amount=amount,
                )

    async with session_factory() as session:
        for reservation_id, _, _ in invalid_inputs:
            row = await session.scalar(
                select(BudgetReservation).where(
                    BudgetReservation.id == reservation_id
                )
            )
            assert row is None
