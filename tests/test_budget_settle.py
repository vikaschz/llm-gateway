from concurrent.futures import ThreadPoolExecutor

import pytest

SETTLED = 1
MARKER_MISSING = 2
COUNTER_MISSING = 3
NEGATIVE = 4
CORRUPT = 5
INVALID_ARGUMENTS = -1


def test_settle_lower_actual_cost(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "settle-lower"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)
    redis_client.set(marker, 100, ex=3600)

    result = settle_script(keys=[counter, marker], args=[60])

    assert result[0] == SETTLED
    assert result[1] == "settled"
    assert int(result[2]) == 60
    assert int(result[3]) == -40

    assert redis_client.get(counter) == "60"
    assert redis_client.exists(marker) == 0


def test_settle_higher_actual_cost(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "settle-higher"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)
    redis_client.set(marker, 100, ex=3600)

    result = settle_script(keys=[counter, marker], args=[150])

    assert result[0] == SETTLED
    assert result[1] == "settled"
    assert int(result[2]) == 150
    assert int(result[3]) == 50

    assert redis_client.get(counter) == "150"
    assert redis_client.exists(marker) == 0


def test_settle_second_call_is_noop(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "settle-twice"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)
    redis_client.set(marker, 100, ex=3600)

    first = settle_script(keys=[counter, marker], args=[70])

    assert first[0] == SETTLED
    assert redis_client.get(counter) == "70"
    assert redis_client.exists(marker) == 0

    second = settle_script(keys=[counter, marker], args=[150])

    assert second[0] == MARKER_MISSING
    assert second[1] == "marker_missing"

    assert redis_client.get(counter) == "70"
    assert redis_client.exists(marker) == 0


def test_release_with_zero_actual_cost(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "release-zero"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 250, ex=3600)
    redis_client.set(marker, 250, ex=3600)

    result = settle_script(keys=[counter, marker], args=[0])

    assert result[0] == SETTLED
    assert result[1] == "settled"
    assert int(result[2]) == 0
    assert int(result[3]) == -250

    assert redis_client.get(counter) == "0"
    assert redis_client.exists(marker) == 0


def test_settle_marker_missing(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "missing-marker"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)

    result = settle_script(keys=[counter, marker], args=[50])

    assert result[0] == MARKER_MISSING
    assert result[1] == "marker_missing"

    assert redis_client.get(counter) == "100"
    assert redis_client.exists(marker) == 0


def test_settle_counter_missing(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "missing-counter"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(marker, 100, ex=3600)

    result = settle_script(keys=[counter, marker], args=[50])

    assert result[0] == COUNTER_MISSING
    assert result[1] == "counter_missing"

    assert redis_client.exists(counter) == 0
    assert redis_client.get(marker) == "100"


def test_settle_rejects_negative_result(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "negative-result"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 50, ex=3600)
    redis_client.set(marker, 100, ex=3600)

    result = settle_script(keys=[counter, marker], args=[0])

    assert result[0] == NEGATIVE
    assert result[1] == "negative"

    assert redis_client.get(counter) == "50"
    assert redis_client.get(marker) == "100"


def test_settle_actual_cost_equal_to_reservation(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "equal-cost"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)
    redis_client.set(marker, 100, ex=3600)

    result = settle_script(keys=[counter, marker], args=[100])

    assert result[0] == SETTLED
    assert result[1] == "settled"
    assert int(result[2]) == 100
    assert int(result[3]) == 0

    assert redis_client.get(counter) == "100"
    assert redis_client.exists(marker) == 0


@pytest.mark.parametrize("actual_cost", ["-1", "abc", ""])
def test_settle_invalid_arguments_change_nothing(
    redis_client,
    settle_script,
    counter_key,
    marker_key,
    actual_cost,
):
    period = "202610"
    reservation_id = f"invalid-{actual_cost!r}"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)
    redis_client.set(marker, 100, ex=3600)

    result = settle_script(keys=[counter, marker], args=[actual_cost])

    assert result[0] == INVALID_ARGUMENTS
    assert result[1] == "invalid_arguments"

    assert redis_client.get(counter) == "100"
    assert redis_client.get(marker) == "100"


def test_settle_corrupt_marker_changes_nothing(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "corrupt-marker"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, 100, ex=3600)
    redis_client.set(marker, "not-an-integer", ex=3600)

    result = settle_script(keys=[counter, marker], args=[50])

    assert result[0] == CORRUPT
    assert result[1] == "corrupt_marker"

    assert redis_client.get(counter) == "100"
    assert redis_client.get(marker) == "not-an-integer"


def test_settle_corrupt_counter_changes_nothing(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "corrupt-counter"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    redis_client.set(counter, "not-an-integer", ex=3600)
    redis_client.set(marker, "100", ex=3600)

    result = settle_script(keys=[counter, marker], args=[50])

    assert result[0] == CORRUPT
    assert result[1] == "corrupt_counter"

    assert redis_client.get(counter) == "not-an-integer"
    assert redis_client.get(marker) == "100"


def test_concurrent_settlement_applies_delta_exactly_once(
    redis_client, settle_script, counter_key, marker_key
):
    period = "202610"
    reservation_id = "concurrent-settlement"

    counter = counter_key(period)
    marker = marker_key(period, reservation_id)

    # Counter includes the reservation of 100.
    # Settling at 60 should apply a single delta of -40.
    redis_client.set(counter, 200, ex=3600)
    redis_client.set(marker, 100, ex=3600)

    def settle_once():
        return settle_script(keys=[counter, marker], args=[60])

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda _: settle_once(), range(20)))

    statuses = [result[1] for result in results]

    assert statuses.count("settled") == 1
    assert statuses.count("marker_missing") == 19

    successful = next(result for result in results if result[0] == SETTLED)

    assert int(successful[2]) == 160
    assert int(successful[3]) == -40

    assert redis_client.get(counter) == "160"
    assert redis_client.exists(marker) == 0


def test_reserve_then_settle_round_trip(
    redis_client,
    reserve_script,
    settle_script,
    counter_key,
    marker_key,
):
    period = "202610"
    counter = counter_key(period)

    redis_client.set(counter, 0, ex=3600)

    # Reserve 100, then settle at 60.
    reservation_id = "roundtrip-settle"
    marker = marker_key(period, reservation_id)

    reserve_result = reserve_script(
        keys=[counter, marker],
        args=[100, 1000],
    )

    assert reserve_result[0] == 1
    assert redis_client.get(counter) == "100"

    settle_result = settle_script(keys=[counter, marker], args=[60])

    assert settle_result[0] == SETTLED
    assert redis_client.get(counter) == "60"
    assert redis_client.exists(marker) == 0

    # Reserve another 100, then release it with actual cost 0.
    release_id = "roundtrip-release"
    release_marker = marker_key(period, release_id)

    reserve_result = reserve_script(
        keys=[counter, release_marker],
        args=[100, 1000],
    )

    assert reserve_result[0] == 1
    assert redis_client.get(counter) == "160"

    release_result = settle_script(
        keys=[counter, release_marker],
        args=[0],
    )

    assert release_result[0] == SETTLED
    assert redis_client.get(counter) == "60"
    assert redis_client.exists(release_marker) == 0

