import math
from datetime import UTC, datetime

# ============================================================
# Budget configuration
# ============================================================

NANODOLLARS_PER_DOLLAR = 1_000_000_000

BUDGET_LIMIT_NANODOLLARS = 10 * NANODOLLARS_PER_DOLLAR



# openai/gpt-oss-20b:
#   Input:  $0.075 / 1M tokens
#   Output: $0.30  / 1M tokens
#
# Therefore:
#   $0.075 / 1M = 75 nanodollars/token
#   $0.30  / 1M = 300 nanodollars/token
MODEL_PRICES_NANODOLLARS = {
    "openai/gpt-oss-20b": {
        "input": 75,
        "output": 300,
    },
}

# Groq limits for openai/gpt-oss-20b.
MODEL_LIMITS = {
    "openai/gpt-oss-20b": {
        "context_tokens": 131_072,
        "max_completion_tokens": 65_536,
    },
}

# Gateway deliberately imposes a lower completion cap.
DEFAULT_MAX_COMPLETION_TOKENS = 8_192
MAX_COMPLETION_TOKENS_CAP = 16_384

BUDGET_KEY_PREFIX = "budget:committed:"
BUDGET_TTL_GRACE_SECONDS = 300


# ============================================================
# Model validation
# ============================================================

def get_model_prices(model: str) -> dict[str, int]:
    try:
        prices = MODEL_PRICES_NANODOLLARS[model]
    except KeyError:
        raise ValueError(
            f"Unsupported budget model: {model}"
        ) from None

    for name, price in prices.items():
        if (
            not isinstance(price, int)
            or isinstance(price, bool)
            or price <= 0
        ):
            raise ValueError(
                f"Invalid {name} price for model {model}"
            )

    return prices


def get_model_limits(model: str) -> dict[str, int]:
    try:
        return MODEL_LIMITS[model]
    except KeyError:
        raise ValueError(
            f"Unsupported budget model: {model}"
        ) from None


# ============================================================
# Token / reservation validation
# ============================================================

def validate_token_counts(*counts: int) -> None:
    if any(
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for count in counts
    ):
        raise ValueError(
            "Token counts must be non-negative integers"
        )


def validate_reservation_amount(reservation: int) -> None:
    if (
        not isinstance(reservation, int)
        or isinstance(reservation, bool)
    ):
        raise ValueError(  # noqa: TRY004
            "Reservation must be an integer"
        )

    if reservation <= 0:
        raise ValueError(
            "Budget reservation must be greater than zero"
        )


def validate_max_completion_tokens(
    max_completion_tokens: int,
) -> None:
    validate_token_counts(max_completion_tokens)

    if max_completion_tokens > MAX_COMPLETION_TOKENS_CAP:
        raise ValueError(
            "max_completion_tokens exceeds gateway cap"
        )


# ============================================================
# Cost calculation
# ============================================================

def estimate_reservation(
    model: str,
    prompt_tokens: int,
    max_completion_tokens: int,
) -> int:
    """
    Return the maximum estimated cost in nanodollars.

    Reservation is based on:
        actual prompt tokens
        + requested maximum completion tokens
    """

    validate_token_counts(
        prompt_tokens,
        max_completion_tokens,
    )

    validate_max_completion_tokens(
        max_completion_tokens
    )

    prices = get_model_prices(model)

    reservation = (
        prompt_tokens * prices["input"]
        + max_completion_tokens * prices["output"]
    )

    validate_reservation_amount(reservation)

    return reservation


def calculate_actual_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> int:
    """
    Return actual provider cost in nanodollars
    using reported token usage.
    """

    validate_token_counts(
        prompt_tokens,
        completion_tokens,
    )

    prices = get_model_prices(model)

    actual_cost = (
        prompt_tokens * prices["input"]
        + completion_tokens * prices["output"]
    )

    return actual_cost


# ============================================================
# Budget period / TTL
# ============================================================

def calculate_budget_period(
    now: datetime,
) -> tuple[str, int, datetime]:
    """
    Calculate the monthly budget period using UTC.

    Returns:
        period_id:
            UTC month identifier, e.g. "202610".

        ttl:
            Seconds until the next UTC month boundary,
            plus the configured grace period.

        next_month:
            UTC datetime marking the next month boundary.
    """

    if not isinstance(now, datetime):
        raise TypeError(
            "now must be a datetime"
        )

    if (
        now.tzinfo is None
        or now.utcoffset() is None
    ):
        raise ValueError(
            "now must be timezone-aware"
        )

    # Normalize every input to UTC before calculating
    # the period. This prevents local timezone boundaries
    # from changing the accounting month.
    now = now.astimezone(UTC)

    if now.month == 12:
        next_month = datetime(
            now.year + 1,
            1,
            1,
            tzinfo=UTC,
        )
    else:
        next_month = datetime(
            now.year,
            now.month + 1,
            1,
            tzinfo=UTC,
        )

    period_id = now.strftime("%Y%m")

    seconds_until_boundary = (
        next_month - now
    ).total_seconds()

    ttl = (
        math.ceil(seconds_until_boundary)
        + BUDGET_TTL_GRACE_SECONDS
    )

    return (
        period_id,
        max(1, ttl),
        next_month,
    )

async def get_budget_period_and_ttl(
    redis_client,
) -> tuple[str, int, str, int]:
    seconds, microseconds = await redis_client.time()

    now = datetime.fromtimestamp(
        seconds + microseconds / 1_000_000,
        tz=UTC,
    )

    period_id, ttl, next_month = calculate_budget_period(now)

    expires_at = (
        int(next_month.timestamp())
        + BUDGET_TTL_GRACE_SECONDS
    )

    resets_at = (
        next_month
        .isoformat()
        .replace("+00:00", "Z")
    )

    return period_id, ttl, resets_at, expires_at