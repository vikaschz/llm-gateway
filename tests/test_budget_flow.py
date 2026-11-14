import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from services.budget_redis import ReserveOutcome


@pytest.fixture
def gateway(monkeypatch):
    """
    Import main.py without loading the real embedding model.
    """

    fake_model = Mock()
    fake_model.get_embedding_dimension.return_value = 384
    fake_model.encode.return_value = np.zeros(384, dtype=np.float32)

    fake_transformer = Mock(return_value=fake_model)

    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        fake_transformer,
    )

    sys.modules.pop("main", None)

    main = importlib.import_module("main")

    # Prevent actual cache operations.
    monkeypatch.setattr(
        main.embedding_model,
        "encode",
        Mock(return_value=np.zeros(384, dtype=np.float32)),
    )

    # Mock Redis rate limiting.
    monkeypatch.setattr(
        main,
        "rate_limiter",
        Mock(return_value=[1, 59, 9000]),
    )

    # Mock API key selection.
    monkeypatch.setattr(
        main,
        "select_key_script",
        Mock(return_value=["test-key", 123]),
    )

    monkeypatch.setattr(
        main,
        "get_model_limits",
        Mock(return_value={"max_completion_tokens": 65536}),
    )

    monkeypatch.setattr(
        main,
        "estimate_reservation",
        Mock(return_value=1000),
    )

    monkeypatch.setattr(
        main,
        "get_budget_period_and_ttl",
        AsyncMock(
            return_value=(
                "2026-11",
                100000,
                "2026-12-01T00:00:00Z",
                1796083200,
            )
        ),
    )

    # Mock SQL operations.
    main.create_pending_reservation = AsyncMock()

    main.confirm_reservation = AsyncMock(
        return_value=SimpleNamespace(changed=True)
    )

    main.release_reservation = AsyncMock()

    # Mock Redis budget operations.
    main.ensure_counter = AsyncMock()

    main.reserve_budget = AsyncMock()

    # Mock token rollback.
    main.rollback_global_token_reservation = Mock()

    # Mock key rotation rollback.
    main.rollback_rotation_reservation = Mock()

    return main


class FakeSession:
    def __init__(self):
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeAsyncClient:
    """
    Fake Groq HTTP client.

    Returns HTTP 400 to stop endpoint execution after
    successful budget reservation.
    """

    calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        FakeAsyncClient.calls += 1

        return SimpleNamespace(
            status_code=400,
            headers={},
            text="Test response",
            json=lambda: {"error": "Test response"},
        )


def make_request(main):
    body = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "Test prompt",
            }
        ],
        "max_completion_tokens": 100,
    }

    session = FakeSession()

    app = SimpleNamespace(
        state=SimpleNamespace(
            api_key_config=SimpleNamespace(
                keys_by_model={"test-model": ["test-key"]},
                keys_by_id={"test-key": "fake-api-key"},
            ),
            redis=Mock(),
            async_session_factory=Mock(return_value=session),
        )
    )

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app,
    }

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(body).encode(),
            "more_body": False,
        }

    request = Request(scope, receive)

    return request, session


@pytest.fixture
def fake_provider(monkeypatch, gateway):
    FakeAsyncClient.calls = 0

    monkeypatch.setattr(
        gateway.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    monkeypatch.setattr(
        gateway,
        "MAX_PROVIDER_ATTEMPTS",
        1,
    )


def budget_result(outcome):
    return SimpleNamespace(outcome=outcome)


# ---------------------------------------------------------
# TEST 1: ACCEPTED
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_accepted(gateway, fake_provider):

    gateway.reserve_budget.return_value = budget_result(
        ReserveOutcome.ACCEPTED
    )

    request, session = make_request(gateway)

    with pytest.raises(HTTPException) as exc:
        await gateway.chat_completion(request)

    # HTTP 400 comes from our fake provider response.
    assert exc.value.status_code == 400

    gateway.confirm_reservation.assert_awaited_once()

    gateway.release_reservation.assert_not_awaited()

    gateway.create_pending_reservation.assert_awaited_once()


# ---------------------------------------------------------
# TEST 2: ALREADY_RESERVED
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_already_reserved(gateway, fake_provider):

    gateway.reserve_budget.return_value = budget_result(
        ReserveOutcome.ALREADY_RESERVED
    )

    request, session = make_request(gateway)

    with pytest.raises(HTTPException) as exc:
        await gateway.chat_completion(request)

    assert exc.value.status_code == 400

    gateway.confirm_reservation.assert_awaited_once()

    gateway.release_reservation.assert_not_awaited()


# ---------------------------------------------------------
# TEST 3: EXCEEDED
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_exceeded(gateway, fake_provider):

    gateway.reserve_budget.return_value = budget_result(
        ReserveOutcome.EXCEEDED
    )

    request, session = make_request(gateway)

    with pytest.raises(HTTPException) as exc:
        await gateway.chat_completion(request)

    assert exc.value.status_code == 402

    gateway.release_reservation.assert_awaited_once()

    gateway.confirm_reservation.assert_not_awaited()

    gateway.rollback_global_token_reservation.assert_called_once()

    # Groq must not be called.
    assert FakeAsyncClient.calls == 0


# ---------------------------------------------------------
# TEST 4: RECOVERY_REQUIRED -> ACCEPTED
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_recovery_then_success(gateway, fake_provider):

    gateway.reserve_budget.side_effect = [
        budget_result(ReserveOutcome.RECOVERY_REQUIRED),
        budget_result(ReserveOutcome.ACCEPTED),
    ]

    request, session = make_request(gateway)

    with pytest.raises(HTTPException) as exc:
        await gateway.chat_completion(request)

    assert exc.value.status_code == 400

    assert gateway.reserve_budget.await_count == 2

    assert gateway.ensure_counter.await_count == 2

    gateway.confirm_reservation.assert_awaited_once()

    gateway.release_reservation.assert_not_awaited()


# ---------------------------------------------------------
# TEST 5: RECOVERY_REQUIRED -> FAILURE
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_recovery_failed(gateway, fake_provider):

    gateway.reserve_budget.side_effect = [
        budget_result(ReserveOutcome.RECOVERY_REQUIRED),
        budget_result(ReserveOutcome.CORRUPT),
    ]

    request, session = make_request(gateway)

    with pytest.raises(HTTPException) as exc:
        await gateway.chat_completion(request)

    assert exc.value.status_code == 503

    assert gateway.reserve_budget.await_count == 2

    assert gateway.ensure_counter.await_count == 2

    gateway.release_reservation.assert_awaited_once()

    gateway.confirm_reservation.assert_not_awaited()

    gateway.rollback_global_token_reservation.assert_called_once()

    assert FakeAsyncClient.calls == 0


# ---------------------------------------------------------
# TEST 6: OTHER REDIS FAILURE
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_other_failure(gateway, fake_provider):

    gateway.reserve_budget.return_value = budget_result(
        ReserveOutcome.CORRUPT
    )

    request, session = make_request(gateway)

    with pytest.raises(HTTPException) as exc:
        await gateway.chat_completion(request)

    assert exc.value.status_code == 503

    gateway.release_reservation.assert_awaited_once()

    gateway.confirm_reservation.assert_not_awaited()

    gateway.rollback_global_token_reservation.assert_called_once()

    assert FakeAsyncClient.calls == 0