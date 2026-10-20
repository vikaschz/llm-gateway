from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class BudgetReservation(Base):
    __tablename__ = "budget_reservations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    request_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    period_id: Mapped[str] = mapped_column(
        String(6),
        nullable=False,
    )

    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    key_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    state: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
    )

    reserved_amount: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    actual_cost: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    release_reason: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    reserved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "state IN "
            "('pending', 'reserved', 'settled', 'released', 'uncertain')",
            name="ck_budget_reservation_state",
        ),
        CheckConstraint(
            "reserved_amount >= 0",
            name="ck_budget_reserved_amount_nonnegative",
        ),
        CheckConstraint(
            "actual_cost IS NULL OR actual_cost >= 0",
            name="ck_budget_actual_cost_nonnegative",
        ),
        CheckConstraint(
            "state != 'settled' OR actual_cost IS NOT NULL",
            name="ck_budget_settled_requires_actual_cost",
        ),
        Index(
            "ix_budget_period_state",
            "period_id",
            "state",
        ),
        Index(
            "ix_budget_state_updated",
            "state",
            "updated_at",
        ),
    )