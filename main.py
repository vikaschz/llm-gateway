import logging
import uuid
from contextlib import asynccontextmanager

import faiss
import httpx
import numpy as np

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis as AsyncRedis
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from budget import (
    BUDGET_LIMIT_NANODOLLARS,
    DEFAULT_MAX_COMPLETION_TOKENS,
    estimate_reservation,
    get_budget_period_and_ttl,
    get_model_limits,
    validate_max_completion_tokens,
)
from config import load_api_keys
from database import Base, DATABASE_URL, SessionLocal
from key_rotation import seed_and_reconcile
from models import RequestLog, SemanticCache
from services.budget_accounting import (
    confirm_reservation,
    create_pending_reservation,
    release_reservation,
)
from services.budget_redis import (
    ReserveOutcome,
    ensure_counter,
    load_budget_scripts,
    reserve_budget,
)

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import faiss
import httpx
import numpy as np
import redis
import tiktoken

logger = logging.getLogger(__name__)
# ---------------------------------------------------------
# Redis connection
# ---------------------------------------------------------

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

# ---------------------------------------------------------
# Key rotation Lua scripts
# ---------------------------------------------------------

with open("rotation_select_key.lua", "r") as f:
    select_key_script = r.register_script(f.read())

with open("rotation_true_up.lua", "r") as f:
    true_up_script = r.register_script(f.read())

with open("rotation_mark_failed.lua", "r") as f:
    mark_failed_script = r.register_script(f.read())


# ---------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    # -----------------------------------------------------
    # Async Redis client
    # -----------------------------------------------------

    async_redis = AsyncRedis(
        host="localhost",
        port=6379,
        decode_responses=True,
    )

    # -----------------------------------------------------
    # Async SQLAlchemy engine and session factory
    # -----------------------------------------------------

    async_database_url = DATABASE_URL.replace(
        "sqlite:///",
        "sqlite+aiosqlite:///",
        1,
    )

    async_engine = create_async_engine(
        async_database_url,
    )

    async_session_factory = async_sessionmaker(
        bind=async_engine,
        expire_on_commit=False,
    )

    # Store resources on application state
    app.state.redis = async_redis
    app.state.async_engine = async_engine
    app.state.async_session_factory = async_session_factory

    try:

        # -------------------------------------------------
        # Register budget model before creating tables
        # -------------------------------------------------

        from services.budget_reservation import BudgetReservation  # noqa: F401

        # -------------------------------------------------
        # Create all database tables
        # -------------------------------------------------

        async with async_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        # -------------------------------------------------
        # Load budget Lua scripts
        # -------------------------------------------------

        app.state.budget_scripts = load_budget_scripts(
            async_redis
        )

        # -------------------------------------------------
        # Rebuild FAISS index after tables exist
        # -------------------------------------------------

        rebuild_faiss_index()

        # -------------------------------------------------
        # Load API key configuration
        # -------------------------------------------------

        api_key_config = load_api_keys()

        # -------------------------------------------------
        # Initialize key rotation
        # -------------------------------------------------

        seed_summary = seed_and_reconcile(
            r,
            api_key_config,
        )

        print("API key rotation initialized:")
        print(seed_summary)

        app.state.api_key_config = api_key_config

        yield

    finally:

        # -------------------------------------------------
        # Close async Redis client
        # -------------------------------------------------

        await async_redis.aclose()

        # -------------------------------------------------
        # Dispose async SQLAlchemy engine
        # -------------------------------------------------

        await async_engine.dispose()

        # -------------------------------------------------
        # Close synchronous Redis client
        # -------------------------------------------------

        r.close()


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Groq OpenAI-compatible API endpoint
# ---------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ---------------------------------------------------------
# Provider retry configuration
# ---------------------------------------------------------

MAX_PROVIDER_ATTEMPTS = 3

# ---------------------------------------------------------
# Semantic cache / FAISS
# ---------------------------------------------------------


def rebuild_faiss_index():
    global faiss_index, cache_ids

    db = SessionLocal()

    try:

        # Load all cached records from database
        cache_records = db.query(SemanticCache).all()

        embeddings = []
        cache_ids = []

        # Read stored embeddings
        for record in cache_records:

            # Skip records without an embedding
            if record.embedding is None:
                continue

            embedding = np.frombuffer(
                record.embedding,
                dtype=np.float32,
            )

            embeddings.append(embedding)
            cache_ids.append(record.id)

        # Stack embeddings into a 2D matrix
        if embeddings:

            embedding_matrix = np.vstack(embeddings)

            # Add embeddings to FAISS
            faiss_index.add(embedding_matrix)

    finally:
        db.close()


# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Get embedding dimension
embedding_dimension = embedding_model.get_embedding_dimension()


# Create empty FAISS index
faiss_index = faiss.IndexFlatIP(embedding_dimension)


# FAISS vector position -> database/cache ID
cache_ids = []

# ---------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------

# Load Lua script used for atomic rate limiting
with open("rate_limit.lua") as f:
    lua_source = f.read()


# Register Lua script with Redis
rate_limiter = r.register_script(lua_source)


# ---------------------------------------------------------
# Token counting configuration
# ---------------------------------------------------------

encoding = tiktoken.get_encoding("o200k_base")


# ---------------------------------------------------------
# Request rate limit configuration
# ---------------------------------------------------------

REQ_CAPACITY = 60
REQ_RATE = 60


# ---------------------------------------------------------
# Token rate limit configuration
# ---------------------------------------------------------

TOK_CAPACITY = 10000
TOK_RATE = 10000


# ---------------------------------------------------------
# Prompt token overhead
# ---------------------------------------------------------

PROMPT_OVERHEAD_BASE = 66
PROMPT_OVERHEAD_PER_MESSAGE = 3


# ---------------------------------------------------------
# Retry-After helper
# ---------------------------------------------------------


def parse_retry_after(response: httpx.Response) -> int:
    """
    Parse Retry-After from the provider response.

    If missing or invalid, use the configured fallback.
    The Lua script independently clamps this to 1-300 seconds.
    """

    retry_after = response.headers.get("Retry-After")

    if retry_after is None:
        return 30

    try:
        return int(retry_after)
    except ValueError:
        return 30


# ---------------------------------------------------------
# Rotation reservation rollback
# ---------------------------------------------------------


def rollback_rotation_reservation(
    model: str,
    window_id: int,
    key_id: str,
    reservation_tokens: int,
):
    """
    Roll back a rotation reservation when the request
    produced no known token usage.

    rotation_true_up.lua expects:

        delta = actual - reserved

    Therefore:

        actual = 0
        delta  = -reserved
    """

    result = true_up_script(
        args=[
            model,
            window_id,
            key_id,
            -reservation_tokens,
        ]
    )

    return result


# ---------------------------------------------------------
# Global rate-limit token rollback
# ---------------------------------------------------------


def rollback_global_token_reservation(
    reservation_tokens: int,
):
    """
    The rate limiter reserves token capacity before the
    provider request.

    If the provider request fails before producing known
    usage, return that reserved capacity.
    """

    r.hincrby(
        "rate_limit:global",
        "tok_level",
        reservation_tokens,
    )


# ---------------------------------------------------------
# Mark failed key
# ---------------------------------------------------------


def mark_failed(
    key_id: str,
    model: str,
    failure_type: str,
    retry_after: int = 0,
):
    """
    Update key-rotation state through rotation_mark_failed.lua.
    """

    return mark_failed_script(
        args=[
            key_id,
            model,
            failure_type,
            retry_after,
        ]
    )


# ---------------------------------------------------------
# Chat completion endpoint
# ---------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completion(request: Request):

    body = await request.json()

    # Access FastAPI application state
    api_key_config = request.app.state.api_key_config

    # -----------------------------------------------------
    # Model validation
    # -----------------------------------------------------

    model = body.get("model")

    if model not in api_key_config.keys_by_model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not configured",
        )

    try:
        model_limits = get_model_limits(model)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # -----------------------------------------------------
    # Completion-token validation
    # -----------------------------------------------------

    # Modern parameter takes precedence over legacy parameter.
    if "max_completion_tokens" in body:
        max_completion_tokens = body["max_completion_tokens"]

    elif "max_tokens" in body:
        max_completion_tokens = body["max_tokens"]

    else:
        max_completion_tokens = DEFAULT_MAX_COMPLETION_TOKENS

    # Validate integer type.
    # bool is also an int subclass in Python.
    if (
        not isinstance(max_completion_tokens, int)
        or isinstance(max_completion_tokens, bool)
    ):
        raise HTTPException(
            status_code=400,
            detail="max_completion_tokens must be an integer",
        )

    # Validate gateway completion cap.
    try:
        validate_max_completion_tokens(max_completion_tokens)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Validate model-specific completion limit.
    if max_completion_tokens > model_limits["max_completion_tokens"]:
        raise HTTPException(
            status_code=400,
            detail="Completion tokens exceed model limit",
        )

    # Normalize request body.
    # The provider receives exactly the limit used
    # for budget estimation.
    body.pop("max_tokens", None)

    body["max_completion_tokens"] = max_completion_tokens

    # -----------------------------------------------------
    # Extract messages
    # -----------------------------------------------------

    messages = body.get(
        "messages",
        [],
    )

    text = []

    for message in messages:

        role = message.get(
            "role",
            "",
        )

        content = message.get(
            "content",
            "",
        )

        text.append(role)
        text.append(content)

    joined_text = "\n".join(text)

    # -----------------------------------------------------
    # Semantic cache lookup
    # -----------------------------------------------------

    # Create embedding once.
    # Reuse it for both FAISS lookup and cache insertion.
    embedding = embedding_model.encode(
        joined_text,
        normalize_embeddings=True,
    )

    # FAISS expects a 2D query vector
    query_vector = np.array(
        [embedding],
        dtype=np.float32,
    )

    # Find nearest cached embedding
    scores, indices = faiss_index.search(
        query_vector,
        k=1,
    )

    SIMILARITY_THRESHOLD = 0.85

    print(
        "FAISS score:",
        scores[0][0],
    )

    print(
        "FAISS index size:",
        faiss_index.ntotal,
    )

    if faiss_index.ntotal > 0 and scores[0][0] >= SIMILARITY_THRESHOLD:

        # FAISS position
        index = indices[0][0]

        # FAISS position -> database ID
        cache_id = cache_ids[index]

        db = SessionLocal()

        try:

            cached_record = (
                db.query(SemanticCache).filter(SemanticCache.id == cache_id).first()
            )

            if cached_record:

                cached_data = json.loads(cached_record.response)

                # Log cache hit
                log = RequestLog(
                    model=cached_data.get("model"),
                    key_id= None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost=0,
                    source="cache",
                )

                db.add(log)
                db.commit()

                return cached_data

        finally:
            db.close()

    # -----------------------------------------------------
    # Token reservation
    # -----------------------------------------------------

    # Estimate prompt tokens.
    padding = (
        PROMPT_OVERHEAD_BASE
        + PROMPT_OVERHEAD_PER_MESSAGE * len(messages)
    )

    prompt_tokens_estimate = (
        len(encoding.encode(joined_text)) + padding
    )

    # Reserve prompt + maximum completion tokens.
    token_reservation = (
        prompt_tokens_estimate + max_completion_tokens
    )

    print(
        f"reservation={token_reservation} "
        f"prompt_est={prompt_tokens_estimate} "
        f"max_completion={max_completion_tokens}"
    )

    # -----------------------------------------------------
    # Atomic Redis rate-limit check
    # -----------------------------------------------------

    result = rate_limiter(
        keys=[],
        args=[
            REQ_CAPACITY,
            REQ_RATE,
            TOK_CAPACITY,
            TOK_RATE,
            1,
            token_reservation,
        ],
    )

    allowed = int(result[0])
    req_level = result[1]
    tok_level = result[2]

    if allowed == 0:

        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "request_tokens_remaining": req_level,
                "token_budget_remaining": tok_level,
            },
        )

    # -----------------------------------------------------
    # Provider request / key rotation
    # -----------------------------------------------------

    for attempt in range(MAX_PROVIDER_ATTEMPTS):

        print(f"Provider attempt " f"{attempt + 1}/" f"{MAX_PROVIDER_ATTEMPTS}")

        # -------------------------------------------------
        # Select API key and reserve rotation tokens
        # -------------------------------------------------

        selection = select_key_script(
            args=[
                model,
                token_reservation,
            ]
        )

        # -------------------------------------------------
        # All keys temporarily cooling
        # -------------------------------------------------

        if selection[0] == "COOLDOWN":

            recovery_time = int(selection[1])

            retry_after = max(
                1,
                recovery_time - int(time.time()),
            )

            # The selection itself already created
            # a rotation reservation only when a key
            # was actually selected, so there is
            # nothing to roll back here.

            # IMPORTANT:
            # We must also return the global token
            # reservation because no provider request
            # will be made.
            rollback_global_token_reservation(token_reservation)

            raise HTTPException(
                status_code=503,
                detail=("All configured API keys " "are temporarily unavailable"),
                headers={"Retry-After": str(retry_after)},
            )

        # -------------------------------------------------
        # All keys permanently dead
        # -------------------------------------------------

        if selection[0] == "ALL_KEYS_DEAD":

            rollback_global_token_reservation(token_reservation)

            raise HTTPException(
                status_code=503,
                detail=("All configured API keys " "are unavailable"),
            )

        # -------------------------------------------------
        # Successful key selection
        # -------------------------------------------------

        key_id = selection[0]
        window_id = int(selection[1])

        print(f"key_id: {key_id}")
        print(f"window_id: {window_id}")

        api_key = api_key_config.keys_by_id.get(key_id)

        if api_key is None:
            rollback_rotation_reservation(
                model,
                window_id,
                key_id,
                token_reservation,
            )

            if attempt + 1 < MAX_PROVIDER_ATTEMPTS:
                continue

            rollback_global_token_reservation(token_reservation)

            raise HTTPException(
                status_code=503,
                detail="Selected API key is unavailable",
            )

        # -------------------------------------------------
        # Monthly budget reservation
        # -------------------------------------------------

        reservation_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        async_redis = request.app.state.redis
        session_factory = request.app.state.async_session_factory

        estimated_cost = estimate_reservation(
            model,
            prompt_tokens_estimate,
            max_completion_tokens,
        )

        period_id, ttl, resets_at, expires_at = (
            await get_budget_period_and_ttl(async_redis)
        )

        async with session_factory() as session:

            # Create pending database record first.
            await create_pending_reservation(
                session=session,
                reservation_id=reservation_id,
                request_id=request_id,
                period_id=period_id,
                model=model,
                reserved_amount=estimated_cost,
                key_id=key_id,
            )

            await session.commit()

            # Ensure the monthly Redis counter is initialized.
            await ensure_counter(
                redis=async_redis,
                session=session,
                period_id=period_id,
                expires_at=expires_at,
            )

            # Atomically reserve estimated monetary cost.
            budget_result = await reserve_budget(
                redis=async_redis,
                period_id=period_id,
                reservation_id=reservation_id,
                amount=estimated_cost,
                limit=BUDGET_LIMIT_NANODOLLARS,
            )

            if budget_result.outcome not in (
                ReserveOutcome.ACCEPTED,
                ReserveOutcome.ALREADY_RESERVED,
            ):

                await release_reservation(
    session,
    reservation_id,
    "budget_exceeded",
)

                await session.commit()

                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Monthly budget reservation rejected",
                        "budget_period": period_id,
                        "budget_resets_at": resets_at,
                    },
                )

            # Redis reservation accepted.
            transition_result = await confirm_reservation(
                session,
                reservation_id,
            )

            if not transition_result.changed:
                raise RuntimeError(
                    "Redis accepted budget reservation, "
                    "but database confirmation failed"
                )

            await session.commit()

        print(f"Using API key: {key_id}")

        # -------------------------------------------------
        # Groq API request
        # -------------------------------------------------

        headers = {
            "Authorization": (f"Bearer {api_key}"),
            "Content-Type": ("application/json"),
        }

        try:

            async with httpx.AsyncClient() as client:

                response = await client.post(
                    GROQ_URL,
                    headers=headers,
                    json=body,
                    timeout=60.0,
                )

        except httpx.TimeoutException:

            print(f"Provider timeout " f"for key={key_id}")

            # IMPORTANT:
            # We do NOT roll back the rotation
            # reservation because the actual provider
            # usage is unknown.
            #
            # We also do NOT mark the key as failed.
            #
            # Retry once according to the timeout policy.
            if attempt == 0 and attempt + 1 < MAX_PROVIDER_ATTEMPTS:

                continue

            raise HTTPException(
                status_code=504,
                detail="Upstream provider timeout",
            )

        except httpx.RequestError as exc:

            print(f"Provider network error " f"for key={key_id}: {exc}")

            # Network failure means we do not know
            # whether Groq received/processed the request.
            #
            # Keep the rotation reservation.
            if attempt + 1 < MAX_PROVIDER_ATTEMPTS:

                continue

            raise HTTPException(
                status_code=502,
                detail="Upstream provider request failed",
            )

        # -------------------------------------------------
        # Convert response to dictionary
        # -------------------------------------------------

        try:
            data = response.json()
        except ValueError:
            data = {"error": response.text}

        status_code = response.status_code

        print(
            "Groq status:",
            status_code,
        )

        # -------------------------------------------------
        # Success
        # -------------------------------------------------

        if status_code == 200:

            # -------------------------------------------------
            # Extract actual token usage
            # -------------------------------------------------

            prompt_tokens = data["usage"]["prompt_tokens"]

            completion_tokens = data["usage"]["completion_tokens"]

            total_tokens = data["usage"]["total_tokens"]

            response_model = data["model"]

            # -------------------------------------------------
            # Calculate request cost
            # -------------------------------------------------

            prompt_price = 0.075 / 1_000_000

            completion_price = 0.30 / 1_000_000

            cost = prompt_tokens * prompt_price + completion_tokens * completion_price # type: ignore

            # -------------------------------------------------
            # Rotation true-up
            # -------------------------------------------------

            actual_total = total_tokens

            # rotation_true_up.lua expects:
            #
            # delta = actual - reserved

            rotation_delta = actual_total - token_reservation

            print(
                f"rotation reserved="
                f"{token_reservation} "
                f"actual={actual_total} "
                f"delta={rotation_delta}"
            )

            if rotation_delta != 0:

                true_up_result = true_up_script(
                    args=[
                        model,
                        window_id,
                        key_id,
                        rotation_delta,
                    ]
                )

                print(
                    "Rotation true-up:",
                    true_up_result,
                )

            # -------------------------------------------------
            # Global rate-limit reconciliation
            # -------------------------------------------------

            # Existing rate limiter uses:
            #
            # delta = reserved - actual

            global_delta = token_reservation - actual_total

            if global_delta != 0:

                r.hincrby(
                    "rate_limit:global",
                    "tok_level",
                    global_delta,
                )

            # -------------------------------------------------
            # Store request information
            # -------------------------------------------------

            db = SessionLocal()

            try:

                log = RequestLog(
                    model=response_model,
                    key_id =key_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                    source="groq",
                )

                db.add(log)
                db.commit()

                # -------------------------------------------------
                # Add response to semantic cache
                # -------------------------------------------------

                if data["choices"][0].get("finish_reason") == "stop":

                    embedding_bytes = embedding.astype(np.float32).tobytes()

                    cache_entry = SemanticCache(
                        prompt=joined_text,
                        embedding=embedding_bytes,
                        response=json.dumps(data),
                    )

                    db.add(cache_entry)
                    db.commit()

                    db.refresh(cache_entry)

                    # Add embedding to FAISS
                    vector = np.array(
                        [embedding],
                        dtype=np.float32,
                    )

                    faiss_index.add(vector)

                    # Maintain FAISS -> database ID mapping
                    cache_ids.append(cache_entry.id)

            finally:
                db.close()

            return data

        # -------------------------------------------------
        # 401 Unauthorized
        # -------------------------------------------------

        if status_code == 401:

            # Provider did not successfully consume
            # the reserved request in our accounting.
            rollback_rotation_reservation(
                model,
                window_id,
                key_id,
                token_reservation,
            )

            mark_result = mark_failed(
                key_id,
                model,
                "401",
                0,
            )

            print(
                "Mark failed:",
                mark_result,
            )

            # Retry using another key.
            if attempt + 1 < MAX_PROVIDER_ATTEMPTS:

                continue

            rollback_global_token_reservation(token_reservation)

            raise HTTPException(
                status_code=401,
                detail=data,
            )

        # -------------------------------------------------
        # 429 Too Many Requests
        # -------------------------------------------------

        if status_code == 429:

            rollback_rotation_reservation(
                model,
                window_id,
                key_id,
                token_reservation,
            )

            retry_after = parse_retry_after(response)

            mark_result = mark_failed(
                key_id,
                model,
                "429",
                retry_after,
            )

            print(
                "Mark failed:",
                mark_result,
            )

            # Retry with another key.
            if attempt + 1 < MAX_PROVIDER_ATTEMPTS:

                continue

            rollback_global_token_reservation(token_reservation)

            raise HTTPException(
                status_code=429,
                detail=data,
                headers={"Retry-After": str(retry_after)},
            )

        # -------------------------------------------------
        # Client/provider errors that should not retry
        # -------------------------------------------------

        if status_code in (
            400,
            403,
            413,
            422,
        ):

            rollback_rotation_reservation(
                model,
                window_id,
                key_id,
                token_reservation,
            )

            rollback_global_token_reservation(token_reservation)

            # 403 does not mark the key dead because
            # the problem may be model/key permissions.
            #
            # 400/413/422 are client-side request errors.
            # Do not rotate/retry them.

            raise HTTPException(
                status_code=status_code,
                detail=data,
            )

        # -------------------------------------------------
        # 5xx / 498 provider failures
        # -------------------------------------------------

        if status_code in (
            498,
            500,
            502,
            503,
        ):

            rollback_rotation_reservation(
                model,
                window_id,
                key_id,
                token_reservation,
            )

            if attempt + 1 < MAX_PROVIDER_ATTEMPTS:

                continue

            rollback_global_token_reservation(token_reservation)

            raise HTTPException(
                status_code=502,
                detail=data,
            )

        # -------------------------------------------------
        # Unknown provider response
        # -------------------------------------------------

        rollback_rotation_reservation(
            model,
            window_id,
            key_id,
            token_reservation,
        )

        rollback_global_token_reservation(token_reservation)

        raise HTTPException(
            status_code=status_code,
            detail=data,
        )

    # Should never be reached.
    raise HTTPException(
        status_code=503,
        detail="Provider request failed",
    )
