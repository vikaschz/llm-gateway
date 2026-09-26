from contextlib import asynccontextmanager

import faiss
import httpx
import json
import numpy as np
import redis
import tiktoken

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from config import load_api_keys
from database import Base, engine, SessionLocal
from key_rotation import seed_and_reconcile
from models import RequestLog, SemanticCache

# ---------------------------------------------------------
# Redis connection
# ---------------------------------------------------------

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)


# ---------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load API key configuration from .env
    api_key_config = load_api_keys()

    # Initialize and reconcile Redis key-rotation state
    seed_summary = seed_and_reconcile(
        r,
        api_key_config,
    )

    print("API key rotation initialized:")
    print(seed_summary)

    # Store configuration on the FastAPI application
    # so request handlers can access it.
    app.state.api_key_config = api_key_config

    yield


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
# Create database tables
# ---------------------------------------------------------

Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------
# Semantic cache / FAISS
# ---------------------------------------------------------


def rebuild_faiss_index():
    global faiss_index, cache_ids

    db = SessionLocal()

    try:

        # Load all cached records from the database
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


# Rebuild FAISS index from SQLite
rebuild_faiss_index()


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
# Chat completion endpoint
# ---------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completion(request: Request):

    body = await request.json()

    # Access FastAPI application state
    api_key_config = request.app.state.api_key_config

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

    padding = PROMPT_OVERHEAD_BASE + (PROMPT_OVERHEAD_PER_MESSAGE * len(messages))

    prompt_tokens_estimate = len(encoding.encode(joined_text)) + padding

    max_completion_tokens = body.get(
        "max_completion_tokens",
        1024,
    )

    token_reservation = prompt_tokens_estimate + max_completion_tokens

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

    model = body.get("model")

    if model not in api_key_config.keys_by_model:

        raise HTTPException(
            status_code=400,
            detail=(f"Model '{model}' " "is not configured"),
        )

    key_id = api_key_config.keys_by_model[model][0]

    api_key = api_key_config.keys_by_id.get(key_id)

    if api_key is None:

        raise HTTPException(
            status_code=503,
            detail="Configured API key is unavailable",
        )

    print(f"Using API key: {key_id}")

    # -----------------------------------------------------
    # Groq API request
    # -----------------------------------------------------

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:

        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=body,
        )

        # Convert response to dictionary
        data = response.json()

        print(
            "Groq status:",
            response.status_code,
        )

        if response.status_code != 200:

            raise HTTPException(
                status_code=response.status_code,
                detail=data,
            )

        # -------------------------------------------------
        # Extract actual token usage
        # -------------------------------------------------

        prompt_tokens = data["usage"]["prompt_tokens"]

        completion_tokens = data["usage"]["completion_tokens"]

        total_tokens = data["usage"]["total_tokens"]

        model = data["model"]

        # -------------------------------------------------
        # Calculate request cost
        # -------------------------------------------------

        prompt_price = 0.075 / 1_000_000

        completion_price = 0.30 / 1_000_000

        cost = prompt_tokens * prompt_price + completion_tokens * completion_price

        # -------------------------------------------------
        # Reconcile reserved tokens
        # -------------------------------------------------

        actual_total = data["usage"]["total_tokens"]

        delta = token_reservation - actual_total

        print(f"actual_total={actual_total} " f"delta={delta}")

        if delta != 0:

            r.hincrby(
                "rate_limit:global",
                "tok_level",
                delta,
            )

        # -------------------------------------------------
        # Store request information
        # -------------------------------------------------

        db = SessionLocal()

        try:

            log = RequestLog(
                model=model,
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
