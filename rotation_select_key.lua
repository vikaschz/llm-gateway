local model = ARGV[1]

local redis_time = redis.call("TIME")

local now = tonumber(redis_time[1])

local cooldown_key = "rotation:cooldown:" .. model
local available_key = "rotation:available:" .. model

local expired_keys = redis.call(
    "ZRANGEBYSCORE",
    cooldown_key,
    "-inf",
    now
)

for _, key_id in ipairs(expired_keys) do

    redis.call(
        "ZREM",
        cooldown_key,
        key_id
    )

    redis.call(
        "SADD",
        available_key,
        key_id
    )

end

local candidates = redis.call(
    "SMEMBERS",
    available_key
)

local healthy_candidates = {}

for _, key_id in ipairs(candidates) do

    local health = redis.call(
        "HGET",
        "rotation:key:" .. key_id,
        "health"
    )

    if health == "healthy" then
        table.insert(
            healthy_candidates,
            key_id
        )
    end

end

local WINDOW_SECONDS = 60

local window_id = math.floor(
    now / WINDOW_SECONDS
)

local usage_key =
    "rotation:usage:" .. model .. ":" .. window_id

local best_key = nil
local best_usage = nil
local best_last_selected_at = nil

for _, key_id in ipairs(healthy_candidates) do

    local usage = redis.call(
        "ZSCORE",
        usage_key,
        key_id
    )

    usage = tonumber(usage) or 0

    local last_selected_at = redis.call(
        "HGET",
        "rotation:key:" .. key_id,
        "last_selected_at"
    )

    last_selected_at = tonumber(
        last_selected_at
    ) or 0

    if best_key == nil
        or usage < best_usage
        or (
            usage == best_usage
            and last_selected_at < best_last_selected_at
        )
        or (
            usage == best_usage
            and last_selected_at == best_last_selected_at
            and key_id < best_key
        )
    then
        best_key = key_id
        best_usage = usage
        best_last_selected_at = last_selected_at
    end

end

-- Step 8: Handle empty case

if #healthy_candidates == 0 then

    if redis.call(
        "ZCARD",
        cooldown_key
    ) > 0 then

        local earliest = redis.call(
            "ZRANGE",
            cooldown_key,
            0,
            0,
            "WITHSCORES"
        )

        return {
            "COOLDOWN",
            tonumber(earliest[2])
        }

    end

    return {
        "ALL_KEYS_DEAD"
    }

end

-- Step 9: Commit selection

local reservation_tokens = tonumber(ARGV[2])

redis.call(
    "ZINCRBY",
    usage_key,
    reservation_tokens,
    best_key
)

redis.call(
    "EXPIRE",
    usage_key,
    300,
    "NX"
)

redis.call(
    "HSET",
    "rotation:key:" .. best_key,
    "last_selected_at",
    now
)

return {
    best_key,
    window_id
}