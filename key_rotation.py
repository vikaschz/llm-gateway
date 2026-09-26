import redis

from config import APIKeyConfig


HEALTH_HEALTHY = "healthy"


def seed_and_reconcile(
    r: redis.Redis,
    
    config: APIKeyConfig,
) -> dict:
    """
    Additive seeding + reconciliation against current config.

    Returns a summary dict for the startup log.
    """

    initialized_keys = set()
    removed_keys = set()
    models_seeded = set()

    # -----------------------------------------------------
    # 1. Additive seeding
    # -----------------------------------------------------

    # A key can belong to multiple models, but its
    # key-level Redis hash only needs initialization once.
    initialized_key_ids = set()

    for model, key_ids in config.keys_by_model.items():

        available_key = f"rotation:available:{model}"
        models_seeded.add(model)

        for key_id in key_ids:

            if key_id not in initialized_key_ids:

                key_hash = f"rotation:key:{key_id}"

                # Never overwrite existing health.
                if r.hsetnx(
                    key_hash,
                    "health",
                    HEALTH_HEALTHY,
                ):
                    initialized_keys.add(key_id)

                # Never overwrite existing selection history.
                r.hsetnx(
                    key_hash,
                    "last_selected_at",
                    0,
                )

                initialized_key_ids.add(key_id)

            # SADD is idempotent.
            r.sadd(
                available_key,
                key_id,
            )

    # -----------------------------------------------------
    # 2. Reconciliation
    # -----------------------------------------------------

    for model, key_ids in config.keys_by_model.items():

        current_key_ids = set(key_ids)

        available_key = f"rotation:available:{model}"
        cooldown_key = f"rotation:cooldown:{model}"

        available_ids = r.smembers(available_key)
        cooldown_ids = r.zrange(cooldown_key, 0, -1)

        redis_key_ids = available_ids | set(cooldown_ids)

        stale_key_ids = redis_key_ids - current_key_ids

        for key_id in stale_key_ids:

            r.srem(
                available_key,
                key_id,
            )

            r.zrem(
                cooldown_key,
                key_id,
            )

            removed_keys.add(key_id)

    # -----------------------------------------------------
    # 3. Intentionally do not scan Redis for old hashes.
    #
    # rotation:key:<key_id> hashes belonging to removed
    # configuration entries may remain.
    #
    # They are harmless because:
    #
    #   - current model pools are reconciled
    #   - selection checks key health
    #   - request code uses keys_by_id.get(key_id)
    # -----------------------------------------------------

    return {
        "keys_initialized": sorted(initialized_keys),
        "keys_removed": sorted(removed_keys),
        "models_seeded": sorted(models_seeded),
    }