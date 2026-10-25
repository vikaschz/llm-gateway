"""Async Redis wrapper for monthly budget reservation scripts."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from services.budget_accounting import (
    get_rebuild_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESERVE_SCRIPT_PATH = PROJECT_ROOT / "budget_reserve.lua"
SETTLE_SCRIPT_PATH = PROJECT_ROOT / "budget_settle.lua"


def counter_key(period_id: str) -> str:
    """Return the Redis key for a month's committed budget."""
    return f"budget:committed:{period_id}"


def marker_key(period_id: str, reservation_id: str) -> str:
    """Return the idempotency marker key for one reservation."""
    return f"budget:marker:{period_id}:{reservation_id}"


class ReserveOutcome(Enum):
    ACCEPTED = 1
    ALREADY_RESERVED = 2
    RECOVERY_REQUIRED = 3
    CORRUPT = 4
    OVERSIZED = 5
    EXCEEDED = 6
    NO_TTL = 7
    INVALID_ARGUMENTS = -1


class SettleOutcome(Enum):
    SETTLED = 1
    MARKER_MISSING = 2
    COUNTER_MISSING = 3
    NEGATIVE = 4
    CORRUPT = 5
    INVALID_ARGUMENTS = -1


@dataclass(frozen=True)
class BudgetScriptResult:
    outcome: ReserveOutcome | SettleOutcome
    committed: int | None = None
    limit: int | None = None
    remaining: int | None = None
    delta: int | None = None


@dataclass(frozen=True)
class BudgetScripts:
    reserve: Any
    settle: Any


def load_budget_scripts(redis: Redis) -> BudgetScripts:
    """Register both Lua scripts using paths rooted at the project directory."""
    reserve_source = RESERVE_SCRIPT_PATH.read_text(encoding="utf-8")
    settle_source = SETTLE_SCRIPT_PATH.read_text(encoding="utf-8")

    return BudgetScripts(
        reserve=redis.register_script(reserve_source),
        settle=redis.register_script(settle_source),
    )


def _get_scripts(redis: Redis) -> BudgetScripts:
    """Cache registered scripts on this Redis client."""
    scripts = getattr(redis, "_budget_scripts", None)

    if scripts is None:
        scripts = load_budget_scripts(redis)
        redis._budget_scripts = scripts

    return scripts


def _integer(value: Any) -> int:
    return int(value)


def _reserve_result(raw: list[Any]) -> BudgetScriptResult:
    try:
        outcome = ReserveOutcome(_integer(raw[0]))
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Unknown budget reserve return code: {raw!r}"
        ) from exc

    values = raw[2:]

    return BudgetScriptResult(
        outcome=outcome,
        committed=_integer(values[0]) if len(values) > 0 else None,
        limit=_integer(values[1]) if len(values) > 1 else None,
        remaining=_integer(values[2]) if len(values) > 2 else None,
    )


def _settle_result(raw: list[Any]) -> BudgetScriptResult:
    try:
        outcome = SettleOutcome(_integer(raw[0]))
    except (ValueError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Unknown budget settle return code: {raw!r}"
        ) from exc

    values = raw[2:]

    return BudgetScriptResult(
        outcome=outcome,
        committed=_integer(values[0]) if len(values) > 0 else None,
        delta=_integer(values[1]) if len(values) > 1 else None,
    )


async def reserve_budget(
    redis: Redis,
    period_id: str,
    reservation_id: str,
    amount: int,
    limit: int,
) -> BudgetScriptResult:
    """Atomically reserve estimated cost against the monthly budget."""
    script = _get_scripts(redis).reserve

    raw = await script(
        keys=[
            counter_key(period_id),
            marker_key(period_id, reservation_id),
        ],
        args=[amount, limit],
    )

    return _reserve_result(raw)


async def settle_budget(
    redis: Redis,
    period_id: str,
    reservation_id: str,
    actual_cost: int,
) -> BudgetScriptResult:
    """Settle a reservation to actual cost and remove its marker."""
    script = _get_scripts(redis).settle

    raw = await script(
        keys=[
            counter_key(period_id),
            marker_key(period_id, reservation_id),
        ],
        args=[actual_cost],
    )

    return _settle_result(raw)


async def release_budget(
    redis: Redis,
    period_id: str,
    reservation_id: str,
) -> BudgetScriptResult:
    """Release a reservation by settling it at zero actual cost."""
    return await settle_budget(
        redis,
        period_id,
        reservation_id,
        0,
    )


async def ensure_counter(
    redis: Redis,
    session: AsyncSession,
    period_id: str,
    expires_at: int,
    lock_ttl_ms: int = 10_000,
    wait_timeout: float = 12.0,
) -> bool:
    """
    Ensure the Redis budget counter and its outstanding markers exist.

    The database snapshot is taken only after acquiring the per-period
    rebuild lock. Markers are published before the counter, making the
    counter the publication point.
    """
    counter = counter_key(period_id)
    lock_key = f"budget:rebuild-lock:{period_id}"

    # Fast path: skip rebuilding if the counter already exists.
    if await redis.exists(counter):
        return False

    token = secrets.token_urlsafe(32)

    while True:
        # Check counter status before attempting lock acquisition.
        if await redis.exists(counter):
            return False

        acquired = await redis.set(
            lock_key,
            token,
            nx=True,
            px=lock_ttl_ms,
        )

        if not acquired:
            deadline = (
                asyncio.get_running_loop().time() + wait_timeout
            )

            while asyncio.get_running_loop().time() < deadline:
                if await redis.exists(counter):
                    return False

                if not await redis.exists(lock_key):
                    break

                await asyncio.sleep(0.05)

            if await redis.exists(counter):
                return False

            # If the lock still exists after the deadline, raise TimeoutError.
            if await redis.exists(lock_key):
                raise TimeoutError(
                    f"Timed out waiting for lock '{lock_key}' to clear"
                )

            # The lock was released or expired without creating the counter.
            # Retry lock acquisition.
            continue

        try:
            # Double-check existence after acquiring the lock.
            if await redis.exists(counter):
                return False

            total, outstanding = await get_rebuild_snapshot(session, period_id)

        

            # Publish markers first.
            for reservation_id, amount in outstanding:
                await redis.set(
                    marker_key(period_id, reservation_id),
                    amount,
                    nx=True,
                    exat=expires_at,
                )

            # Publish the counter last.
            published = await redis.set(
                counter,
                total,
                nx=True,
                exat=expires_at,
            )

            return bool(published)

        finally:
            # Release the lock only if its token still matches.
            await redis.eval(
                """
                if redis.call("GET", KEYS[1]) == ARGV[1] then
                    return redis.call("DEL", KEYS[1])
                end
                return 0
                """,
                1,
                lock_key,
                token,
            )