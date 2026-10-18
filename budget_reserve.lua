-- KEYS[1] = budget:committed:<YYYYMM>
--
-- ARGV[1] = reservation amount (nanodollars)
-- ARGV[2] = monthly budget limit (nanodollars)
-- ARGV[3] = TTL in seconds
--
-- Return codes:
--   1  = reservation accepted
--   0  = budget exceeded
--  -1  = invalid arguments
--
-- Return format:
--   {status, committed, limit, remaining}
--
-- Guarantees:
--   - Atomic budget check and reservation.
--   - Rejected requests never create or modify the key.
--   - TTL is set only when a new key is created.
--   - Existing key TTL is never refreshed.

local key = KEYS[1]

local reservation = tonumber(ARGV[1])
local budget_limit = tonumber(ARGV[2])
local ttl_seconds = tonumber(ARGV[3])

-- Validate arguments before touching Redis.
if not reservation
    or not budget_limit
    or not ttl_seconds
    or reservation <= 0
    or budget_limit < 0
    or ttl_seconds <= 0
    or reservation ~= math.floor(reservation)
    or budget_limit ~= math.floor(budget_limit)
    or ttl_seconds ~= math.floor(ttl_seconds) then
    return {-1, 0, budget_limit or 0, budget_limit or 0}
end

-- Check whether the monthly key already exists.
local existed = redis.call("EXISTS", key)

local current = 0

if existed == 1 then
    current = tonumber(redis.call("GET", key))

    if not current or current < 0 or current ~= math.floor(current) then
        return {-1, 0, budget_limit, budget_limit}
    end
end

-- Calculate the proposed counter without modifying Redis.
local proposed = current + reservation
local remaining = math.max(0, budget_limit - current)

-- Reject before writing anything.
if proposed > budget_limit then
    return {0, current, budget_limit, remaining}
end

-- Reserve the amount atomically.
redis.call("INCRBY", key, reservation)

-- Set TTL only when the key did not exist before this reservation.
if existed == 0 then
    redis.call("EXPIRE", key, ttl_seconds)
end

local new_remaining = math.max(0, budget_limit - proposed)

return {1, proposed, budget_limit, new_remaining}