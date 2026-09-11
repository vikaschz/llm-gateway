from fastapi import FastAPI
import httpx
from config import GROQ_API_KEY
from database import Base, engine
from models import RequestLog
from database import SessionLocal

app = FastAPI()

Base.metadata.create_all(bind=engine)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.post("/v1/chat/completions")
async def chat_completion(request: dict):
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
