-- KEYS[1] = budget:committed:<YYYYMM>
--
-- ARGV[1] = original reservation amount (nanodollars)
-- ARGV[2] = actual cost (nanodollars)
--
-- Return codes:
--   1  = true-up successful
--   0  = budget key missing; do not recreate it
--  -1  = invalid arguments
--
-- Return format:
--   {status, committed, delta}
--
-- delta = actual_cost - reservation
--   Negative: release the excess reservation
--   Positive: add the additional cost

local key = KEYS[1]

local reservation = tonumber(ARGV[1])
local actual_cost = tonumber(ARGV[2])

-- Validate arguments before touching Redis.
if not reservation
    or not actual_cost
    or reservation <= 0
    or actual_cost < 0
    or reservation ~= math.floor(reservation)
    or actual_cost ~= math.floor(actual_cost) then
    return {-1, 0, 0}
end

-- Do not recreate a missing monthly key.
if redis.call("EXISTS", key) == 0 then
    return {0, 0, actual_cost - reservation}
end

local current = tonumber(redis.call("GET", key))

-- Reject corrupted counter values.
if not current
    or current < 0
    or current ~= math.floor(current) then
    return {-1, 0, 0}
end

local delta = actual_cost - reservation
local updated = current + delta

-- Prevent negative budget accounting.
if updated < 0 then
    return {-1, current, delta}
end

-- Apply only the difference.
if delta ~= 0 then
    redis.call("INCRBY", key, delta)
end

-- Intentionally do not modify the existing TTL.
return {1, updated, delta}