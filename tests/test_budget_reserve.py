from concurrent.futures import ThreadPoolExecutor  # needed for the concurrency test

import pytest

# Return codes from budget_reserve.lua
ACCEPTED = 1
ALREADY_RESERVED = 2
RECOVERY_REQUIRED = 3
CORRUPT = 4
OVERSIZED = 5
EXCEEDED = 6
NO_TTL = 7
INVALID_ARGUMENTS = -1


def test_reserve_accepted(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "reservation-1"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[50, 1000],
    )

    assert result[0] == ACCEPTED
    assert result[1] == "accepted"
    assert int(result[2]) == 150
    assert int(result[3]) == 1000
    assert int(result[4]) == 850

    assert redis_client.get(counter) == "150"
    assert redis_client.get(marker) == "50"
    assert redis_client.ttl(marker) > 0
    assert redis_client.pexpiretime(marker) <= redis_client.pexpiretime(counter)

def test_reserve_already_reserved(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "reservation-2"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)
    redis_client.set(marker, 50, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[50, 1000],
    )

    assert result[0] == ALREADY_RESERVED
    assert result[1] == "already_reserved"

    assert redis_client.get(counter) == "100"
    assert redis_client.get(marker) == "50"


def test_reserve_requires_recovery_when_counter_missing(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "reservation-3"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    result = reserve_script(
        keys=[counter, marker],
        args=[50, 1000],
    )

    assert result[0] == RECOVERY_REQUIRED
    assert result[1] == "recovery_required"

    assert redis_client.exists(counter) == 0
    assert redis_client.exists(marker) == 0


def test_reserve_rejects_corrupt_counter(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "reservation-4"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, "not-an-integer", ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[50, 1000],
    )

    assert result[0] == CORRUPT
    assert result[1] == "corrupt"

    assert redis_client.get(counter) == "not-an-integer"
    assert redis_client.exists(marker) == 0


def test_reserve_rejects_oversized_reservation(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "reservation-5"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[1500, 1000],
    )

    assert result[0] == OVERSIZED
    assert result[1] == "oversized"
    assert int(result[2]) == 100
    assert int(result[3]) == 1000
    assert int(result[4]) == 900

    assert redis_client.get(counter) == "100"
    assert redis_client.exists(marker) == 0


def test_reserve_rejects_when_budget_would_be_exceeded(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "reservation-6"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 900, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[200, 1000],
    )

    assert result[0] == EXCEEDED
    assert result[1] == "exceeded"
    assert int(result[2]) == 900
    assert int(result[3]) == 1000
    assert int(result[4]) == 100

    assert redis_client.get(counter) == "900"
    assert redis_client.exists(marker) == 0


# ---- Add the new tests below (you write these) ----



def test_reserve_exact_fit(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "exact-fit"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 900, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[100, 1000],
    )

    assert result[0] == ACCEPTED
    assert result[1] == "accepted"
    assert int(result[2]) == 1000
    assert int(result[3]) == 1000
    assert int(result[4]) == 0

    assert redis_client.get(counter) == "1000"
    assert redis_client.get(marker) == "100"


def test_reserve_at_limit_rejects_additional_amount(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "at-limit"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 1000, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[1, 1000],
    )

    assert result[0] == EXCEEDED
    assert result[1] == "exceeded"
    assert int(result[2]) == 1000
    assert int(result[3]) == 1000
    assert int(result[4]) == 0

    assert redis_client.get(counter) == "1000"
    assert redis_client.exists(marker) == 0


def test_reserve_equal_to_limit(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "equal-to-limit"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 0, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[1000, 1000],
    )

    assert result[0] == ACCEPTED
    assert result[1] == "accepted"
    assert int(result[2]) == 1000
    assert int(result[3]) == 1000
    assert int(result[4]) == 0

    assert redis_client.get(counter) == "1000"
    assert redis_client.get(marker) == "1000"


@pytest.mark.parametrize("amount", ["0", "-5", "abc"])
def test_reserve_invalid_amount_changes_nothing(
    redis_client,
    reserve_script,
    counter_key,
    marker_key,
    amount,
):
    period = "202610"
    reservation_id = f"invalid-{amount}"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)

    result = reserve_script(
        keys=[counter, marker],
        args=[amount, 1000],
    )

    assert result[0] == INVALID_ARGUMENTS
    assert result[1] == "invalid_arguments"

    assert redis_client.get(counter) == "100"
    assert redis_client.exists(marker) == 0


def test_reserve_rejects_counter_without_expiry(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "no-expiry"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    # No expiry: Redis TTL returns -1.
    redis_client.set(counter, 100)

    result = reserve_script(
        keys=[counter, marker],
        args=[50, 1000],
    )

    assert result[0] == NO_TTL
    assert result[1] == "no_ttl"

    assert redis_client.get(counter) == "100"
    assert redis_client.exists(marker) == 0


def test_concurrent_reservations_are_atomic(
    redis_client, reserve_script, counter_key, marker_key
):
    period = "202610"
    counter = counter_key(period)
    limit = 1000
    amount = 100

    redis_client.set(counter, 0, ex=3600)

    def reserve_once(i):
        reservation_id = f"c-{i}"
        marker = marker_key(period, reservation_id)

        return reserve_script(
            keys=[counter, marker],
            args=[amount, limit],
        )

    # Fifty unique reservations compete for a limit of 1000.
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(reserve_once, range(50)))

    accepted = [result for result in results if result[0] == ACCEPTED]
    exceeded = [result for result in results if result[0] == EXCEEDED]

    assert len(accepted) == 10
    assert len(exceeded) == 40

    assert redis_client.get(counter) == "1000"

    markers = redis_client.keys(f"budget:marker:{period}:*")
    assert len(markers) == 10

    for marker in markers:
        assert redis_client.get(marker) == "100"