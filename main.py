from fastapi import FastAPI
import httpx
from config import GROQ_API_KEY
from database import Base, engine
from models import RequestLog
from database import SessionLocal
import redis
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tiktoken

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Base.metadata.create_all(bind=engine)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


r = redis.Redis(host="localhost", port=6379, decode_responses=True)

with open("rate_limit.lua") as f:
    lua_source = f.read()

rate_limiter = r.register_script(lua_source)


REQ_CAPACITY = 60
REQ_RATE = 60

TOK_CAPACITY = 10000
TOK_RATE = 10000


@app.post("/v1/chat/completions")
async def chat_completion(request: dict):

    messages = request.get("messages", [])

    encoding = tiktoken.get_encoding("o200k_base")

    text = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        text.append(role)
        text.append(content)

    joined_text = "\n".join(text)

    prompt_tokens_estimate = len(encoding.encode(joined_text))

    max_completion_tokens = request.get("max_completion_tokens", 1024)

    token_reservation = prompt_tokens_estimate + max_completion_tokens
    print(f"reservation={token_reservation} prompt_est={prompt_tokens_estimate} max_completion={max_completion_tokens}")

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

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:

        response = await client.post(
            # fill in url, headers, json body
            GROQ_URL,
            headers=headers,
            json=request,
        )
        data = response.json()

        prompt_tokens = data["usage"]["prompt_tokens"]
        completion_tokens = data["usage"]["completion_tokens"]
        total_tokens = data["usage"]["total_tokens"]
        model = data["model"]

        prompt_price = 0.075 / 1_000_000
        completion_price = 0.30 / 1_000_000

        cost = prompt_tokens * prompt_price + completion_tokens * completion_price

        actual_total = data["usage"]["total_tokens"]
        delta = token_reservation - actual_total
        print(f"actual_total={actual_total} delta={delta}")

        if delta != 0:
            r.hincrby("rate_limit:global", "tok_level", delta)

        db = SessionLocal()
        try:
            log = RequestLog(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
            )
            db.add(log)
            db.commit()
        finally:
            db.close()

    return data
