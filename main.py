from fastapi import FastAPI
from config import GROQ_API_KEY
from database import Base, engine
from models import RequestLog, SemanticCache
from database import SessionLocal
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import httpx
import redis
import tiktoken
import numpy as np
import json

app = FastAPI()


# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Groq OpenAI-compatible API endpoint
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# Create database tables if they do not already exist
Base.metadata.create_all(bind=engine)


def rebuild_faiss_index():
    global faiss_index, cache_ids

    db = SessionLocal()

    try:
        # 1. Load all cached records from the database
        cache_records = db.query(SemanticCache).all()

        embeddings = []
        cache_ids = []

        # 2. Read the already-stored embeddings
        for record in cache_records:

            # Skip records that don't have an embedding
            if record.embedding is None:
                continue

            embedding = np.frombuffer(
                record.embedding, dtype=np.float32  #type: ignore
            )  # type: ignore

            embeddings.append(embedding)
            cache_ids.append(record.id)

        # 3. Stack 1D embeddings into a 2D matrix
        if embeddings:
            embedding_matrix = np.vstack(embeddings)

            # 4. Add embeddings to FAISS
            faiss_index.add(embedding_matrix)

    finally:
        # 5. Close the database session
        db.close()


# Load the embedding model once when the application starts
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Get the embedding dimension from the model
embedding_dimension = embedding_model.get_embedding_dimension()


# Create an empty FAISS index
# IndexFlatIP uses inner product, which we can use for cosine similarity
# when embeddings are normalized.
faiss_index = faiss.IndexFlatIP(embedding_dimension)  # type: ignore


# Python list mapping:
# FAISS vector position -> database/cache ID
cache_ids = []


rebuild_faiss_index()


# ---------------------------------------------------------
# Redis connection
# ---------------------------------------------------------

# Connect to Redis running locally
r = redis.Redis(host="localhost", port=6379, decode_responses=True)


# Load the Lua script used for atomic rate limiting
with open("rate_limit.lua") as f:
    lua_source = f.read()


# Register the Lua script with Redis
rate_limiter = r.register_script(lua_source)


# ---------------------------------------------------------
# Token counting configuration
# ---------------------------------------------------------

# Tokenizer used to estimate prompt tokens
encoding = tiktoken.get_encoding("o200k_base")


# ---------------------------------------------------------
# Request rate limit configuration
# ---------------------------------------------------------

# Maximum number of requests allowed in the bucket
REQ_CAPACITY = 60

# Request tokens refilled per time window
REQ_RATE = 60


# ---------------------------------------------------------
# Token rate limit configuration
# ---------------------------------------------------------

# Maximum token budget available in the bucket
TOK_CAPACITY = 10000

# Number of tokens refilled per time window
TOK_RATE = 10000


# ---------------------------------------------------------
# Prompt token overhead
# ---------------------------------------------------------

# Base overhead added to the estimated prompt tokens
PROMPT_OVERHEAD_BASE = 66

# Additional overhead added for every message
PROMPT_OVERHEAD_PER_MESSAGE = 3


# ---------------------------------------------------------
# Chat completion endpoint
# ---------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completion(request: dict):

    # Extract messages from the incoming request
    messages = request.get("messages", [])

    text = []

    # Extract role and content from every message
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        text.append(role)
        text.append(content)

    # Combine all roles and message contents into one string
    joined_text = "\n".join(text)

    # ---------------------------------------------------------
    # Semantic cache lookup
    # ---------------------------------------------------------

    # Create the embedding ONCE.
    # This same embedding will be used for:
    # 1. FAISS lookup
    # 2. Storing the new cache entry
    embedding = embedding_model.encode(joined_text, normalize_embeddings=True)

    # FAISS expects a 2D array for a single query
    query_vector = np.array([embedding], dtype=np.float32)

    # Find the nearest cached embedding
    scores, indices = faiss_index.search(query_vector, k=1)

    # Similarity threshold for accepting a cache hit
    SIMILARITY_THRESHOLD = 0.85
    print("FAISS score:", scores[0][0])
    print("FAISS index size:", faiss_index.ntotal)

    if faiss_index.ntotal > 0 and scores[0][0] >= SIMILARITY_THRESHOLD:
        # FAISS returns the position of the matching vector
        index = indices[0][0]

        # Convert FAISS position -> database record ID
        cache_id = cache_ids[index]

        db = SessionLocal()

        try:
            # Fetch the cached response from SQLite
            cached_record = (
                db.query(SemanticCache).filter(SemanticCache.id == cache_id).first()
            )

            if cached_record:
                cached_data = json.loads(cached_record.response)  # type: ignore

                # Log the cache hit
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

                # Return cached response directly
                return cached_data

        finally:
            db.close()

    # ---------------------------------------------------------
    # Token reservation
    # ---------------------------------------------------------

    # Estimate additional token overhead caused by the
    # chat message structure
    padding = PROMPT_OVERHEAD_BASE + (PROMPT_OVERHEAD_PER_MESSAGE * len(messages))

    # Estimate the number of tokens in the prompt
    prompt_tokens_estimate = len(encoding.encode(joined_text)) + padding

    # Get the maximum number of completion tokens requested.
    # Use 1024 if the client does not provide the value.
    max_completion_tokens = request.get("max_completion_tokens", 1024)

    # Reserve both prompt tokens and possible completion tokens
    token_reservation = prompt_tokens_estimate + max_completion_tokens

    print(
        f"reservation={token_reservation} "
        f"prompt_est={prompt_tokens_estimate} "
        f"max_completion={max_completion_tokens}"
    )

    # ---------------------------------------------------------
    # Atomic Redis rate-limit check
    # ---------------------------------------------------------

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

    # Extract the result returned by the Lua script
    allowed = int(result[0])
    req_level = result[1]
    tok_level = result[2]

    # Reject the request if either rate limit is exceeded
    if allowed == 0:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "request_tokens_remaining": req_level,
                "token_budget_remaining": tok_level,
            },
        )

    # ---------------------------------------------------------
    # Groq API request
    # ---------------------------------------------------------

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:

        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=request,
        )

        # Convert Groq response into a Python dictionary
        data = response.json()

        print("Groq status:", response.status_code)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=data)

        # -----------------------------------------------------
        # Extract actual token usage
        # -----------------------------------------------------

        prompt_tokens = data["usage"]["prompt_tokens"]
        completion_tokens = data["usage"]["completion_tokens"]
        total_tokens = data["usage"]["total_tokens"]
        model = data["model"]

        # -----------------------------------------------------
        # Calculate request cost
        # -----------------------------------------------------

        prompt_price = 0.075 / 1_000_000
        completion_price = 0.30 / 1_000_000

        cost = prompt_tokens * prompt_price + completion_tokens * completion_price

        # -----------------------------------------------------
        # Reconcile reserved tokens with actual usage
        # -----------------------------------------------------

        actual_total = data["usage"]["total_tokens"]

        # Calculate the difference between the reserved
        # token amount and the actual token usage
        delta = token_reservation - actual_total

        print(f"actual_total={actual_total} " f"delta={delta}")

        # Return unused reserved tokens back to the
        # Redis token bucket
        if delta != 0:
            r.hincrby("rate_limit:global", "tok_level", delta)

        # -----------------------------------------------------
        # Store request information in the database
        # -----------------------------------------------------

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

            # Reuse the embedding created during cache lookup.
          

            # Convert NumPy array to bytes for SQLite
            embedding_bytes = embedding.astype(np.float32).tobytes()

            # Create cache record
            cache_entry = SemanticCache(
                prompt=joined_text,
                embedding=embedding_bytes,
                response=json.dumps(data),
            )

            db.add(cache_entry)
            db.commit()
            db.refresh(cache_entry)

            # Add the same embedding to FAISS
            vector = np.array([embedding], dtype=np.float32)

            faiss_index.add(vector)

            # Keep FAISS position -> database ID mapping
            cache_ids.append(cache_entry.id)

        finally:
            # Always close the database session
            db.close()

    return data
