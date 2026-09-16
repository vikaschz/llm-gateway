local KEY = "rate_limit:global"

local req_capacity = tonumber(ARGV[1])
local req_rate = tonumber(ARGV[2])

local tok_capacity = tonumber(ARGV[3])
local tok_rate = tonumber(ARGV[4])

local req_cost = tonumber(ARGV[5])
local tok_cost = tonumber(ARGV[6])

local PERIOD_US = 60000000

local redis_time = redis.call("TIME")
local now = tonumber(redis_time[1]) * 1000000
                 + tonumber(redis_time[2])

local state = redis.call(
    "HMGET",
    KEY,
    "req_level",
    "req_last_refill",
    "tok_level",
    "tok_last_refill"
)

local req_level = tonumber(state[1])
local req_last = tonumber(state[2])

local tok_level = tonumber(state[3])
local tok_last = tonumber(state[4])


if not req_level or not req_last or not tok_level or not tok_last then

    req_level = req_capacity
    req_last = now

    tok_level = tok_capacity
    tok_last = now

end

local req_elapsed = now - req_last

local req_refill = math.floor(
    req_elapsed * req_rate / PERIOD_US)

if req_refill > 0 then

    local new_req_level = math.min(
        req_capacity,
        req_level + req_refill
    )

    if new_req_level >= req_capacity then

        req_level = req_capacity
        req_last = now

    else

        req_level = new_req_level

        local consumed_time = math.floor(
            req_refill * PERIOD_US / req_rate
        )

        req_last = req_last + consumed_time

    end

end


local tok_elapsed = now - tok_last

local tok_refill = math.floor(
    tok_elapsed * tok_rate / PERIOD_US
)

if tok_refill > 0 then

    local new_tok_level = math.min(
        tok_capacity,
        tok_level + tok_refill
    )

    if new_tok_level >= tok_capacity then

        tok_level = tok_capacity
        tok_last = now

    else

        tok_level = new_tok_level

        local consumed_time = math.floor(
            tok_refill * PERIOD_US / tok_rate
        )

        tok_last = tok_last + consumed_time

    end

end

local allowed =
    req_level >= req_cost
    and tok_level >= tok_cost

if not allowed then

  
    redis.call(
        "HSET",
        KEY,
        "req_level", req_level,
        "req_last_refill", req_last,
        "tok_level", tok_level,
        "tok_last_refill", tok_last
    )

    return {
        0,
        req_level,
        tok_level
    }

end

req_level = req_level - req_cost
tok_level = tok_level - tok_cost

redis.call(
    "HSET",
    KEY,
    "req_level", req_level,
    "req_last_refill", req_last,
    "tok_level", tok_level,
    "tok_last_refill", tok_last
)
return {
    1,
    req_level,
    tok_level
}