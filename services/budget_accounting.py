from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from services.budget_reservation import BudgetReservation
from services.budget_transitions import (
    LEGAL_TRANSITIONS,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class TransitionResult:
    changed: bool
    current_state: str | None
    reason: str


def _allowed_source_states(
    target_state: str,
) -> set[str]:
    return {
        source
        for source, target in LEGAL_TRANSITIONS
        if target == target_state
    }


async def create_pending_reservation(
    session: AsyncSession,
    *,
    reservation_id: str,
    request_id: str,
    period_id: str,
    model: str,
    reserved_amount: int,
    key_id: str | None = None,
) -> BudgetReservation:

    if not isinstance(period_id, str) or len(period_id) != 6:
        raise ValueError(
            "period_id must use YYYYMM format"
        )

    if not period_id.isdigit():
        raise ValueError(
            "period_id must use YYYYMM format"
        )

    if reserved_amount <= 0:
        raise ValueError(
            "Budget reservation must be greater than zero"
        )

    now = utc_now()

    reservation = BudgetReservation(
        id=reservation_id,
        request_id=request_id,
        period_id=period_id,
        model=model,
        key_id=key_id,
        state="pending",
        reserved_amount=reserved_amount,
        actual_cost=None,
        created_at=now,
        updated_at=now,
    )

    session.add(reservation)
    await session.flush()

    return reservation


async def transition_reservation(
    session: AsyncSession,
    reservation_id: str,
    target_state: str,
    *,
    actual_cost: int | None = None,
    release_reason: str | None = None,
) -> TransitionResult:

    if target_state not in {
        target
        for _, target in LEGAL_TRANSITIONS
    }:
        raise ValueError(
            f"Unsupported target state: {target_state}"
        )

    values = {
        "state": target_state,
        "updated_at": utc_now(),
    }

    if target_state == "reserved":
        values["reserved_at"] = utc_now()

    if target_state == "settled":
        if actual_cost is None or actual_cost < 0:
            raise ValueError(
                "Settlement requires a non-negative actual_cost"
            )

        values["actual_cost"] = actual_cost
        values["settled_at"] = utc_now()

    elif actual_cost is not None:
        raise ValueError(
            "actual_cost can only be set during settlement"
        )

    if target_state == "released":
        if not release_reason or not release_reason.strip():
            raise ValueError(
                "Release requires a non-empty reason"
            )

        values["release_reason"] = release_reason

    elif release_reason is not None:
        raise ValueError(
            "release_reason can only be set during release"
        )

    allowed_sources = _allowed_source_states(
        target_state
    )

    if not allowed_sources:
        raise ValueError(
            f"No legal transitions to {target_state}"
        )

    result = await session.execute(
        update(BudgetReservation)
        .where(
            BudgetReservation.id == reservation_id,
            BudgetReservation.state.in_(allowed_sources),
        )
        .values(**values)
        .execution_options(     
            synchronize_session=False
        )
    )

    if result.rowcount == 1:
        return TransitionResult(
            changed=True,
            current_state=target_state,
            reason="transitioned",
        )

    current_state = await session.scalar(
        select(BudgetReservation.state).where(
            BudgetReservation.id == reservation_id
        )
    )

    if current_state is None:
        return TransitionResult(
            changed=False,
            current_state=None,
            reason="not_found",
        )

    if current_state == target_state:
        return TransitionResult(
            changed=False,
            current_state=current_state,
            reason="already_in_target_state",
        )

    return TransitionResult(
        changed=False,
        current_state=current_state,
        reason="conflicting_state",
    )


async def confirm_reservation(
    session: AsyncSession,
    reservation_id: str,
) -> TransitionResult:
    return await transition_reservation(
        session,
        reservation_id,
        "reserved",
    )


async def settle_reservation(
    session: AsyncSession,
    reservation_id: str,
    actual_cost: int,
) -> TransitionResult:

    return await transition_reservation(
        session,
        reservation_id,
        "settled",
        actual_cost=actual_cost,
    )


async def release_reservation(
    session: AsyncSession,
    reservation_id: str,
    reason: str,
) -> TransitionResult:

    return await transition_reservation(
        session,
        reservation_id,
        "released",
        release_reason=reason,
    )


async def mark_reservation_uncertain(
    session: AsyncSession,
    reservation_id: str,
) -> TransitionResult:

    return await transition_reservation(
        session,
        reservation_id,
        "uncertain",
    )


async def get_rebuild_total(
    session: AsyncSession,
    period_id: str,
) -> int:
    query = select(
        func.coalesce(
            func.sum(
                case(
                    (BudgetReservation.state == "settled", BudgetReservation.actual_cost),
                    (BudgetReservation.state.in_(["reserved", "uncertain"]), BudgetReservation.reserved_amount),
                    else_=0,
                )
            ),
            0,
        )
    ).where(BudgetReservation.period_id == period_id)

    result = await session.execute(query)
    return int(result.scalar_one())


def calculate_rebuild_total(
    reservations,
) -> int:
    """
    Reference implementation for rebuild accounting.

    settled             -> actual_cost
    reserved/uncertain  -> reserved_amount
    pending/released    -> zero
    """

    total = 0

    for reservation in reservations:

        if reservation.state in {
            "pending",
            "released",
        }:
            continue

        if reservation.state == "settled":

            if reservation.actual_cost is None:
                raise ValueError(
                    f"Settled reservation {reservation.id} "
                    "has no actual_cost"
                )

            total += reservation.actual_cost

        elif reservation.state in {
            "reserved",
            "uncertain",
        }:
            total += reservation.reserved_amount

        else:
            raise ValueError(
                f"Unknown reservation state: "
                f"{reservation.state}"
            )

    return total


async def get_outstanding_reservations(
    session: AsyncSession,
    period_id: str,
) -> list[tuple[str, int]]:
    result = await session.execute(
        select(
            BudgetReservation.id,
            BudgetReservation.reserved_amount,
        )
        .where(
            BudgetReservation.period_id == period_id,
            BudgetReservation.state.in_(["reserved", "uncertain"]),
        )
    )

    return [
        (reservation_id, int(reserved_amount))
        for reservation_id, reserved_amount in result.all()
    ]




async def get_rebuild_snapshot(session, period_id: str):
    result = await session.execute(
        select(
            BudgetReservation.id,
            BudgetReservation.state,
            BudgetReservation.reserved_amount,
            BudgetReservation.actual_cost,
        ).where(
            BudgetReservation.period_id == period_id,
            BudgetReservation.state.in_(
                ["settled", "reserved", "uncertain"]
            ),
        )
    )

    total = 0
    outstanding = []

    for reservation_id, state, reserved_amount, actual_cost in result:
        if state == "settled":
            total += actual_cost or 0
        else:
            total += reserved_amount
            outstanding.append((reservation_id, reserved_amount))

    return total, outstanding