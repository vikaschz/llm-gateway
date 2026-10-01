local key_id = ARGV[1]
local model = ARGV[2]
local failure_type = ARGV[3]
local retry_after = tonumber(ARGV[4])

local key_hash = "rotation:key:" .. key_id
local available_key = "rotation:available:" .. model
local cooldown_key = "rotation:cooldown:" .. model


-- 401: key is permanently dead
if failure_type == "401" then

    redis.call(
        "HSET",
        key_hash,
        "health",
        "dead"
    )

    redis.call(
        "SREM",
        available_key,
        key_id
    )

    redis.call(
        "ZREM",
        cooldown_key,
        key_id
    )

    return {
        "DEAD"
    }

end

-- 429: temporarily cooldown this key
if failure_type == "429" then

    local health = redis.call(
        "HGET",
        key_hash,
        "health"
    )

    if health == "dead" then

        return {
            "ALREADY_DEAD"
        }

    end

    -- Clamp retry_after to 1-300 seconds
    if retry_after == nil then
        retry_after = 30
    end

    if retry_after < 1 then
        retry_after = 1
    elseif retry_after > 300 then
        retry_after = 300
    end

    local redis_time = redis.call("TIME")
    local now = tonumber(redis_time[1])

    local recovery_time = now + retry_after

    redis.call(
        "SREM",
        available_key,
        key_id
    )

    redis.call(
        "ZADD",
        cooldown_key,
        "GT",
        recovery_time,
        key_id
    )

    return {
        "COOLDOWN",
        recovery_time
    }

end


return {
    "UNKNOWN_FAILURE"
}