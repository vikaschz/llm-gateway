# LLM Gateway

A FastAPI-based LLM gateway built to manage model requests, API key rotation, rate limiting, semantic caching, token usage, and request costs through a unified API.

The project explores the infrastructure behind LLM applications, focusing on efficient request handling, atomic resource management, provider failure recovery, and cost control.

## Overview

Applications integrating large language models often need to manage provider API keys, control request volume, track token consumption, and reduce unnecessary API calls.

This project implements a gateway between clients and the Groq API. It provides an OpenAI-compatible chat completions endpoint and uses Redis, Lua scripts, and SQLAlchemy to manage request processing and usage accounting.

### Key Features

* **OpenAI-compatible API:** Exposes a chat completions endpoint using the familiar OpenAI request format.
* **Atomic rate limiting:** Uses Redis and Lua scripts to enforce request and token limits.
* **API key rotation:** Selects among configured provider API keys based on health and usage.
* **Provider failure handling:** Handles authentication failures, rate limits, cooldowns, timeouts, and selected upstream errors.
* **Semantic caching:** Uses sentence embeddings and FAISS to identify similar prompts and reuse cached responses.
* **Token counting:** Estimates prompt tokens using `tiktoken` before sending requests.
* **Usage and cost tracking:** Records request details, token usage, estimated costs, and request sources in a database.
* **Monthly budget enforcement:** Includes Redis/Lua scripts for atomic budget reservations and cost reconciliation.
* **Automated testing:** Uses pytest to validate application behavior and accounting logic.

## Architecture

```text
                   Client Application
                           |
                           v
                  FastAPI LLM Gateway
                           |
                           v
                   Semantic Cache
                  SentenceTransformers
                          +
                         FAISS
                           |
                    Cache Hit?
                     /       \
                   Yes        No
                    |          |
                    v          v
              Return Cached  Validate Model
                Response          |
                                  v
                         Estimate Token Usage
                                  |
                                  v
                          Redis Rate Limiter
                            (Lua Scripts)
                                  |
                                  v
                         Monthly Budget Check
                                  |
                                  v
                          API Key Selection
                                  |
                                  v
                             Groq API
                                  |
                                  v
                        Reconcile Token Usage
                                  |
                                  v
                         Calculate Request Cost
                                  |
                                  v
                         Persist Request Logs
                                  |
                                  v
                         Update Semantic Cache
```

The diagram describes the intended request flow. The exact integration and enforcement status of individual budget components may evolve as development continues.

## Technology Stack

| Technology           | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| Python               | Core application logic                               |
| FastAPI              | HTTP API and request handling                        |
| Groq API             | LLM inference provider                               |
| HTTPX                | Asynchronous HTTP requests                           |
| Redis                | Rate limiting, API key state, and budget counters    |
| Redis Lua            | Atomic rate-limit and accounting operations          |
| SQLAlchemy           | Database models and persistence                      |
| SQLite               | Local database for request logs and cached responses |
| SentenceTransformers | Semantic embeddings                                  |
| FAISS                | Vector similarity search                             |
| tiktoken             | Prompt token estimation                              |
| pytest               | Automated testing                                    |

## How It Works

### 1. Chat Completions API

The gateway exposes an OpenAI-compatible endpoint:

`POST /v1/chat/completions`

Clients submit a model name and a list of messages. The gateway processes the request and, when necessary, forwards it to Groq.

Example request:

```json
{
  "model": "openai/gpt-oss-20b",
  "messages": [
    {
      "role": "user",
      "content": "Explain how Redis rate limiting works."
    }
  ],
  "max_completion_tokens": 512
}
```

Example using Python:

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [
        {
            "role": "user",
            "content": "Explain how Redis rate limiting works."
        }
    ],
    "max_completion_tokens": 512
}

response = requests.post(url, json=payload, timeout=90)

print(response.status_code)
print(response.json())
```

This example assumes the gateway and its dependencies are running locally.

### 2. Semantic Caching

The gateway converts incoming message content into embeddings using the `all-MiniLM-L6-v2` sentence-transformer model.

FAISS performs similarity searches against previously cached prompts.

The current similarity threshold is `0.85`.

When a sufficiently similar cached entry is found, the gateway can return its stored response without making another provider request.

Cache hits are recorded separately with:

* Request source: `cache`
* Token usage: zero for the new provider request
* Provider cost: zero for the cache-hit request

Responses are added to the cache when the provider response meets the current cache eligibility condition.

Semantic caching can reduce repeated inference calls, but similarity alone does not guarantee that two prompts have identical meaning or require identical answers. Cache correctness is an important area for further improvement.

### 3. Redis Rate Limiting

The gateway uses Redis and Lua scripts to implement atomic request and token rate limiting.

Current configured capacities:

| Limit    | Capacity |       Refill rate |
| -------- | -------: | ----------------: |
| Requests |       60 |     60 per minute |
| Tokens   |   10,000 | 10,000 per minute |

Before contacting the provider, the gateway estimates token consumption and reserves capacity.

After receiving a successful response, it reconciles the reservation with actual token usage.

If a request is rejected, the gateway returns HTTP `429 Too Many Requests`.

Atomic Lua execution prevents concurrent requests from independently passing the same rate-limit check using stale counter values.

These values are configuration defaults in the current implementation, not universal limits for every deployment.

### 4. API Key Rotation

Multiple Groq API keys can be configured and associated with supported models.

Redis stores key availability, health, cooldowns, and usage information. Lua scripts perform key selection and related state transitions.

The implementation supports:

* Selecting an available key based on recorded usage.
* Tracking per-key usage within time windows.
* Marking keys as unavailable after authentication failures.
* Temporarily cooling down keys after provider rate limits.
* Retrying selected provider failures with another key.
* Returning service-unavailable errors when no usable keys remain.

Provider timeouts and network errors require special handling because the gateway may not know whether the provider processed the request.

### 5. Token Usage and Cost Tracking

The gateway estimates prompt tokens before making a provider request and uses provider-reported usage after a successful response.

Successful requests are logged with information such as:

* Model
* API key identifier
* Prompt tokens
* Completion tokens
* Total tokens
* Estimated cost
* Request source

The current implementation uses SQLite through SQLAlchemy for persistence.

The configured pricing in the request-processing code must match the provider's current pricing for the recorded costs to be accurate. Pricing should be treated as configurable production data rather than a permanent constant.

### 6. Monthly Budget Enforcement

The project also includes Redis Lua scripts for monthly budget accounting.

The design uses integer nanodollars to avoid relying on floating-point arithmetic for budget counters.

The budget reservation script is responsible for:

1. Validating the reservation amount and budget limit.
2. Checking the current monthly committed amount.
3. Atomically rejecting reservations that would exceed the configured limit.
4. Reserving capacity when the request is allowed.
5. Preserving the existing monthly counter's expiration.

The budget true-up script reconciles the reservation against actual request cost after processing.

Its accounting delta is:

`delta = actual_cost - reservation`

A negative delta releases excess reserved budget. A positive delta accounts for additional cost.

The scripts also validate counter values and avoid recreating a missing monthly counter during true-up.

**Important:** The existence of these scripts does not by itself establish end-to-end budget enforcement. The reservation, provider request, settlement, persistence, and recovery paths must all be integrated and tested before the budget can be relied on as a production spending limit.

## Project Structure

The main components currently include:

```text
llm-gateway/
├── main.py
├── config.py
├── database.py
├── key_rotation.py
├── models.py
├── requirements.txt
├── rate_limit.lua
├── rotation_select_key.lua
├── rotation_true_up.lua
├── rotation_mark_failed.lua
├── budget_reserve.lua
├── budget_trueup.lua
├── tests/
└── README.md
```

The layout is representative of the project's main components. Additional files and test modules may exist in the repository.

| File                       | Responsibility                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `main.py`                  | FastAPI application, request processing, caching, rate limiting, and provider integration |
| `config.py`                | Environment loading and provider API-key configuration validation                         |
| `database.py`              | SQLAlchemy engine, session factory, and declarative base                                  |
| `key_rotation.py`          | Initializing and reconciling API-key state in Redis                                       |
| `models.py`                | Database models for request logs and semantic cache entries                               |
| `rate_limit.lua`           | Atomic request and token rate limiting                                                    |
| `rotation_select_key.lua`  | Selecting an available provider API key                                                   |
| `rotation_true_up.lua`     | Reconciling reserved key-usage tokens with actual usage                                   |
| `rotation_mark_failed.lua` | Updating API-key failure and cooldown state                                               |
| `budget_reserve.lua`       | Atomic monthly budget reservation                                                         |
| `budget_trueup.lua`        | Reconciling reserved budget against actual cost                                           |
| `requirements.txt`         | Python dependencies                                                                       |
| `tests/`                   | Automated tests                                                                           |

## Getting Started

### Prerequisites

Install the following:

* Python 3.12 or a compatible Python version
* Redis Server
* A Groq API key
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/vikaschz/llm-gateway.git
cd llm-gateway
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root.

Configure your Groq API keys as a JSON array:

```dotenv
GROQ_API_KEYS=[{"key_id":"groq-primary","api_key":"YOUR_GROQ_API_KEY","models":["openai/gpt-oss-20b"]}]
```

Replace the placeholder with your actual key.

To configure multiple keys, add additional objects with unique `key_id` values and the appropriate model lists.

The application validates the configuration during startup.

**Security:** Never commit `.env` files, provider API keys, or other secrets to Git. Use environment variables or a dedicated secrets manager in deployed environments.

### 5. Start Redis

Start Redis Server using your installed Redis service or deployment.

Verify that it is accessible:

```bash
redis-cli ping
```

Expected response:

```text
PONG
```

The current application connects to Redis at `localhost:6379`. Adjust the connection configuration if your Redis instance uses another address.

### 6. Start the Gateway

Run the application from the repository root:

```bash
uvicorn main:app --reload
```

The API will be available at:

`http://localhost:8000`

FastAPI's interactive API documentation is available at:

`http://localhost:8000/docs`

## Running Tests

Run the full test suite:

```bash
pytest -v
```

Run a specific test module:

```bash
pytest -v tests/test_budget.py
```

Run selected budget and accounting regression tests when those modules are present in your checkout:

```bash
pytest -v tests/test_budget.py tests/test_budget_accounting.py
```

Use the test filenames present in your current repository if they differ from these examples.

For tests involving Redis and Lua scripts, ensure the test setup provides the required Redis environment or test doubles.

Passing tests demonstrate the behavior covered by those tests; they do not automatically establish production readiness or prove every failure and concurrency scenario is handled correctly.

## API Reference

### `POST /v1/chat/completions`

Accepts a chat-completion request and returns a provider-compatible response or an HTTP error.

The request body follows the OpenAI-compatible chat completions format.

Common responses include:

| Status | Meaning                                                      |
| ------ | ------------------------------------------------------------ |
| `200`  | Successful response or semantic-cache hit                    |
| `400`  | Model is not configured                                      |
| `401`  | Provider authentication failure after retries                |
| `429`  | Rate limit or provider quota rejection                       |
| `502`  | Upstream provider request failure                            |
| `503`  | No usable provider keys or temporary provider unavailability |
| `504`  | Upstream provider timeout                                    |

The precise response depends on the failure path and current implementation.

## Design Considerations

Several implementation details are important for reliable operation:

* **Atomic operations:** Redis Lua scripts prevent races within individual counter updates.
* **Reservation reconciliation:** Estimated resource usage should be reconciled with actual usage after successful requests.
* **Failure ambiguity:** A timeout does not prove that the provider did not process a request.
* **Cache correctness:** Similarity thresholds need to be evaluated against the application's response requirements.
* **Accounting consistency:** Database and Redis state can diverge if a process crashes between operations. Recovery and reconciliation mechanisms are necessary.
* **Budget enforcement:** A hard spending limit requires safe integration of reservations, settlements, failures, and recovery—not just a counter in Redis.
* **Operational security:** Client authentication, restricted CORS, secret management, and monitoring should be addressed before public deployment.

## Current Development Focus

The project is being developed incrementally, with emphasis on correctness and failure handling.

Areas of active engineering include:

* Reliable monthly budget reservations and settlement.
* Database-backed accounting and valid state transitions.
* Recovery and reconciliation when Redis counters are lost or inconsistent.
* Concurrency and regression testing.
* Handling provider errors without corrupting token or cost accounting.
* Improving the separation of request processing, caching, rate limiting, and accounting responsibilities.

Implementation status can change between commits. Consult the current source code and test suite for the exact behavior of each feature.

## Roadmap

Potential areas for further development include:

* Complete end-to-end monthly budget enforcement.
* Stronger client authentication and authorization.
* Configurable provider pricing and budget limits.
* More comprehensive crash-recovery and reconciliation tests.
* Improved cache invalidation and response-equivalence safeguards.
* Structured logging, metrics, and monitoring.
* Docker-based deployment and reproducible development environments.
* Additional provider integrations.

These items represent development directions, not a claim that they are already implemented.

## Contributing

Contributions and technical feedback are welcome.

A useful contribution should include:

1. A clear description of the problem.
2. A focused implementation.
3. Tests covering the relevant behavior.
4. Regression checks for existing functionality.
5. Documentation updates where necessary.

## License

No license has been specified in this README. Add a `LICENSE` file and update this section before distributing the project under an open-source license.

---

**Author:** [vikaschz](https://github.com/vikaschz)

**Repository:** [vikaschz/llm-gateway](https://github.com/vikaschz/llm-gateway)
