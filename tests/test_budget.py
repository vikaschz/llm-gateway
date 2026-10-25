from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from budget import (
    BUDGET_LIMIT_NANODOLLARS,
    BUDGET_TTL_GRACE_SECONDS,
    DEFAULT_MAX_COMPLETION_TOKENS,
    MAX_COMPLETION_TOKENS_CAP,
    MODEL_LIMITS,
    MODEL_PRICES_NANODOLLARS,
    calculate_actual_cost,
    calculate_budget_period,
    estimate_reservation,
    get_budget_period_and_ttl,
    get_model_limits,
    get_model_prices,
    validate_max_completion_tokens,
    validate_reservation_amount,
    validate_token_counts,
)

MODEL = "openai/gpt-oss-20b"


# ============================================================
# Model configuration
# ============================================================

def test_budget_limit():
    assert BUDGET_LIMIT_NANODOLLARS == 10_000_000_000


def test_model_price():
    assert MODEL_PRICES_NANODOLLARS[MODEL]["input"] == 75
    assert MODEL_PRICES_NANODOLLARS[MODEL]["output"] == 300


def test_model_limits():
    assert MODEL_LIMITS[MODEL]["context_tokens"] == 131_072
    assert MODEL_LIMITS[MODEL]["max_completion_tokens"] == 65_536


def test_default_max_completion_tokens():
    assert DEFAULT_MAX_COMPLETION_TOKENS == 8_192


def test_max_completion_tokens_cap():
    assert MAX_COMPLETION_TOKENS_CAP == 16_384


def test_get_model_prices():
    prices = get_model_prices(MODEL)

    assert prices == {
        "input": 75,
        "output": 300,
    }


def test_get_model_prices_unknown_model():
    with pytest.raises(
        ValueError,
        match="Unsupported budget model",
    ):
        get_model_prices("unknown-model")


def test_get_model_limits():
    limits = get_model_limits(MODEL)

    assert limits == {
        "context_tokens": 131_072,
        "max_completion_tokens": 65_536,
    }


def test_get_model_limits_unknown_model():
    with pytest.raises(
        ValueError,
        match="Unsupported budget model",
    ):
        get_model_limits("unknown-model")


def test_all_model_prices_are_positive_integers():
    for model, prices in MODEL_PRICES_NANODOLLARS.items():
        for price_type, price in prices.items():
            assert isinstance(price, int)
            assert not isinstance(price, bool)
            assert price > 0, (
                f"{model} {price_type} price must be positive"
            )


# ============================================================
# Token validation
# ============================================================

@pytest.mark.parametrize(
    "counts",
    [
        (0,),
        (1,),
        (100, 200),
        (65_536,),
    ],
)
def test_validate_token_counts_accepts_non_negative_integers(
    counts,
):
    validate_token_counts(*counts)


@pytest.mark.parametrize(
    "counts",
    [
        (-1,),
        (1, -1),
        (True,),
        (False,),
        (1.5,),
        ("10",),
        (None,),
    ],
)
def test_validate_token_counts_rejects_invalid_values(
    counts,
):
    with pytest.raises(
        ValueError,
        match="Token counts must be non-negative integers",
    ):
        validate_token_counts(*counts)


# ============================================================
# Reservation validation
# ============================================================

def test_validate_reservation_amount_accepts_positive_integer():
    validate_reservation_amount(1)
    validate_reservation_amount(10_000)


@pytest.mark.parametrize(
    "reservation",
    [
        0,
        -1,
        -100,
        True,
        False,
        1.5,
        "100",
        None,
    ],
)
def test_validate_reservation_amount_rejects_invalid_values(
    reservation,
):
    with pytest.raises(ValueError):
        validate_reservation_amount(reservation)


# ============================================================
# max_completion_tokens validation
# ============================================================

def test_validate_max_completion_tokens_accepts_zero():
    validate_max_completion_tokens(0)


def test_validate_max_completion_tokens_accepts_cap():
    validate_max_completion_tokens(
        MAX_COMPLETION_TOKENS_CAP
    )


def test_validate_max_completion_tokens_rejects_above_cap():
    with pytest.raises(
        ValueError,
        match="exceeds gateway cap",
    ):
        validate_max_completion_tokens(
            MAX_COMPLETION_TOKENS_CAP + 1
        )


def test_validate_max_completion_tokens_rejects_negative():
    with pytest.raises(ValueError):
        validate_max_completion_tokens(-1)


# ============================================================
# Cost calculation
# ============================================================

def test_estimate_reservation():
    # 100 input tokens × 75
    # 1,000 completion tokens × 300
    expected = (
        100 * 75
        + 1_000 * 300
    )

    assert estimate_reservation(
        MODEL,
        100,
        1_000,
    ) == expected


def test_estimate_reservation_zero_tokens():
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        estimate_reservation(
            MODEL,
            0,
            0,
        )


def test_estimate_reservation_rejects_above_gateway_cap():
    with pytest.raises(
        ValueError,
        match="exceeds gateway cap",
    ):
        estimate_reservation(
            MODEL,
            100,
            MAX_COMPLETION_TOKENS_CAP + 1,
        )


def test_estimate_reservation_rejects_negative_prompt_tokens():
    with pytest.raises(ValueError):
        estimate_reservation(
            MODEL,
            -1,
            100,
        )


def test_calculate_actual_cost():
    expected = (
        100 * 75
        + 500 * 300
    )

    assert calculate_actual_cost(
        MODEL,
        100,
        500,
    ) == expected


def test_calculate_actual_cost_zero_completion():
    assert calculate_actual_cost(
        MODEL,
        100,
        0,
    ) == 100 * 75


def test_calculate_actual_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        calculate_actual_cost(
            MODEL,
            100,
            -1,
        )


def test_calculate_actual_cost_dollar_equivalents():
    # 1,000,000 input tokens @ 75 nanodollars/token = 75,000,000 nanodollars ($0.075)
    assert calculate_actual_cost(MODEL, 1_000_000, 0) == 75_000_000

    # 1,000,000 output tokens @ 300 nanodollars/token = 300,000,000 nanodollars ($0.30)
    assert calculate_actual_cost(MODEL, 0, 1_000_000) == 300_000_000


# ============================================================
# Monthly period calculation
# ============================================================

def test_december_rollover():
    now = datetime(
        2026,
        12,
        15,
        12,
        0,
        0,
        tzinfo=UTC,
    )

    period_id, ttl, next_month = (
        calculate_budget_period(now)
    )

    assert period_id == "202612"

    assert next_month == datetime(
        2027,
        1,
        1,
        tzinfo=UTC,
    )

    expected_seconds = (
        next_month - now
    ).total_seconds()

    assert ttl == (
        int(expected_seconds)
        + BUDGET_TTL_GRACE_SECONDS
    )


def test_exactly_at_midnight_month_boundary():
    now = datetime(
        2026,
        11,
        1,
        0,
        0,
        0,
        tzinfo=UTC,
    )

    period_id, ttl, next_month = (
        calculate_budget_period(now)
    )

    assert period_id == "202611"

    assert next_month == datetime(
        2026,
        12,
        1,
        tzinfo=UTC,
    )

    assert ttl == (
        30 * 24 * 60 * 60
        + BUDGET_TTL_GRACE_SECONDS
    )


def test_last_second_of_month():
    now = datetime(
        2026,
        10,
        31,
        23,
        59,
        59,
        tzinfo=UTC,
    )

    period_id, ttl, next_month = (
        calculate_budget_period(now)
    )

    assert period_id == "202610"

    assert next_month == datetime(
        2026,
        11,
        1,
        tzinfo=UTC,
    )

    assert ttl == (
        1
        + BUDGET_TTL_GRACE_SECONDS
    )


def test_first_second_of_new_month():
    now = datetime(
        2026,
        11,
        1,
        0,
        0,
        1,
        tzinfo=UTC,
    )

    period_id, ttl, next_month = (
        calculate_budget_period(now)
    )

    assert period_id == "202611"

    assert next_month == datetime(
        2026,
        12,
        1,
        tzinfo=UTC,
    )

    assert ttl == (
        30 * 24 * 60 * 60
        - 1
        + BUDGET_TTL_GRACE_SECONDS
    )


# ============================================================
# Timezone handling
# ============================================================

def test_non_utc_datetime_is_converted_to_utc():
    # 2026-11-01 00:30 IST
    # = 2026-10-31 19:00 UTC
    #
    # Therefore the budget period must still be October.
    ist = timezone(
        timedelta(hours=5, minutes=30)
    )

    now = datetime(
        2026,
        11,
        1,
        0,
        30,
        0,
        tzinfo=ist,
    )

    period_id, ttl, next_month = (
        calculate_budget_period(now)
    )

    assert period_id == "202610"

    assert next_month == datetime(
        2026,
        11,
        1,
        0,
        0,
        0,
        tzinfo=UTC,
    )

    assert ttl == (
        5 * 60 * 60
        + BUDGET_TTL_GRACE_SECONDS
    )


def test_utc_datetime_and_ist_datetime_same_instant():
    ist = timezone(
        timedelta(hours=5, minutes=30)
    )

    utc_time = datetime(
        2026,
        10,
        31,
        19,
        0,
        0,
        tzinfo=UTC,
    )

    ist_time = datetime(
        2026,
        11,
        1,
        0,
        30,
        0,
        tzinfo=ist,
    )

    utc_result = calculate_budget_period(
        utc_time
    )

    ist_result = calculate_budget_period(
        ist_time
    )

    assert utc_result == ist_result


# ============================================================
# Microsecond handling
# ============================================================

def test_microseconds_use_ceil_for_ttl():
    now = datetime(
        2026,
        10,
        31,
        23,
        59,
        59,
        500_000,
        tzinfo=UTC,
    )

    period_id, ttl, _next_month = (
        calculate_budget_period(now)
    )

    assert period_id == "202610"

    # 0.5 seconds remain until the boundary.
    # ceil(0.5) = 1, so the TTL must include one second
    # before adding the 5-minute grace period.
    assert ttl == (
        1
        + BUDGET_TTL_GRACE_SECONDS
    )


def test_microseconds_do_not_expire_before_boundary():
    now = datetime(
        2026,
        10,
        31,
        23,
        59,
        59,
        999_999,
        tzinfo=UTC,
    )

    period_id, ttl, _next_month = (
        calculate_budget_period(now)
    )

    assert period_id == "202610"

    assert ttl == (
        1
        + BUDGET_TTL_GRACE_SECONDS
    )


# ============================================================
# Invalid datetime input
# ============================================================

def test_rejects_naive_datetime():
    now = datetime(
        2026,
        10,
        31,
        23,
        59,
        59,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        calculate_budget_period(now)


def test_rejects_non_datetime():
    with pytest.raises(
        TypeError,
        match="must be a datetime",
    ):
        calculate_budget_period(
            "2026-10-31"
        )


# ============================================================
# Price resolution
# ============================================================

def test_price_resolution_is_integer_nanodollars():
    for prices in MODEL_PRICES_NANODOLLARS.values():
        for price in prices.values():
            assert isinstance(price, int)
            assert not isinstance(price, bool)


def test_price_resolution_cannot_represent_sub_nanodollar_prices():
    # 1 nanodollar/token is the smallest representable unit.
    #
    # Therefore a price below 1 nanodollar/token cannot be
    # represented exactly by the current integer pricing model.
    #
    # Current models must have prices >= 1.
    for prices in MODEL_PRICES_NANODOLLARS.values():
        for price in prices.values():
            assert price >= 1


@pytest.mark.asyncio
async def test_get_budget_period_and_ttl_returns_all_four_values():
    now = datetime(2026, 10, 31, 19, 0, 0, 500_000, tzinfo=UTC)

    redis_client = AsyncMock()
    redis_client.time.return_value = (int(now.timestamp()), now.microsecond)

    period_id, ttl, resets_at, expires_at = await get_budget_period_and_ttl(
        redis_client
    )

    # One Redis TIME read is the only clock used.
    redis_client.time.assert_awaited_once()

    assert period_id == "202610"
    # 5 hours minus 0.5 s rounds up to 5 hours, plus the grace period.
    assert ttl == 5 * 60 * 60 + BUDGET_TTL_GRACE_SECONDS
    assert resets_at == "2026-11-01T00:00:00Z"
    assert expires_at == (
        int(datetime(2026, 11, 1, tzinfo=UTC).timestamp())
        + BUDGET_TTL_GRACE_SECONDS
    )