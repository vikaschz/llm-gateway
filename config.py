import json
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass
class APIKeyConfig:
    keys_by_id: dict[str, str]
    models_by_key: dict[str, list[str]]
    keys_by_model: dict[str, list[str]]


def load_api_keys() -> APIKeyConfig:
    raw = os.getenv("GROQ_API_KEYS")

    if not raw:
        raise RuntimeError("GROQ_API_KEYS is not configured")

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GROQ_API_KEYS contains invalid JSON"
        ) from exc

    if not isinstance(config, list):
        raise RuntimeError(
            "GROQ_API_KEYS must be a JSON array"
        )

    keys_by_id: dict[str, str] = {}
    models_by_key: dict[str, list[str]] = {}
    keys_by_model: dict[str, list[str]] = {}

    for entry in config:

        # Entry must be an object
        if not isinstance(entry, dict):
            raise RuntimeError(
                "Each API key entry must be an object"
            )

        key_id = entry.get("key_id")
        api_key = entry.get("api_key")
        models = entry.get("models")

        # Required fields
        if key_id is None:
            raise RuntimeError(
                "Each key must have key_id"
            )

        if api_key is None:
            raise RuntimeError(
                f"Key '{key_id}' must have api_key"
            )

        if models is None:
            raise RuntimeError(
                f"Key '{key_id}' must have models"
            )

        # Type validation
        if not isinstance(key_id, str):
            raise RuntimeError(
                "key_id must be a string"
            )

        if not isinstance(api_key, str):
            raise RuntimeError(
                f"api_key for '{key_id}' must be a string"
            )

        if not isinstance(models, list):
            raise RuntimeError(
                f"models for '{key_id}' must be a list"
            )

        # Strip strings
        key_id = key_id.strip()
        api_key = api_key.strip()

        # Empty value validation
        if not key_id:
            raise RuntimeError(
                "key_id cannot be empty"
            )

        if not api_key:
            raise RuntimeError(
                f"api_key for '{key_id}' cannot be empty"
            )

        if not models:
            raise RuntimeError(
                f"Key '{key_id}' must have a non-empty models list"
            )

        # Every model must be a string
        if not all(isinstance(model, str) for model in models):
            raise RuntimeError(
                f"All models for '{key_id}' must be strings"
            )

        # Strip model names
        models = [model.strip() for model in models]

        # Model names cannot be empty
        if any(not model for model in models):
            raise RuntimeError(
                f"Models for '{key_id}' cannot contain empty strings"
            )

        # No duplicate models for the same key
        if len(models) != len(set(models)):
            raise RuntimeError(
                f"Key '{key_id}' contains duplicate models"
            )

        # No duplicate key IDs
        if key_id in keys_by_id:
            raise RuntimeError(
                f"Duplicate key_id: {key_id}"
            )

        # key_id -> api_key
        keys_by_id[key_id] = api_key

        # key_id -> models
        models_by_key[key_id] = models

        # model -> key_ids
        for model in models:
            keys_by_model.setdefault(model, []).append(key_id)

    return APIKeyConfig(
        keys_by_id=keys_by_id,
        models_by_key=models_by_key,
        keys_by_model=keys_by_model,
    )