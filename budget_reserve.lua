-- Atomic budget reservation.
--
-- KEYS[1] = budget:committed:<period>
-- KEYS[2] = budget:marker:<period>:<reservation_id>
--
-- ARGV[1] = reservation amount (non-negative integer string; must be > 0)
-- ARGV[2] = budget limit (positive integer string)
--
-- Return codes:
--  1 = ACCEPTED
--  2 = ALREADY_RESERVED
--  3 = RECOVERY_REQUIRED
--  4 = CORRUPT
--  5 = OVERSIZED
--  6 = EXCEEDED
--  7 = NO_TTL
-- -1 = INVALID_ARGUMENTS

local counter_key = KEYS[1]
local marker_key = KEYS[2]

local reservation = ARGV[1]
local limit = ARGV[2]

local MAX_SAFE_INTEGER = 9007199254740991

local function is_non_negative_integer(value)
    return value ~= nil
        and string.match(value, "^%d+$") ~= nil
end

local function parse_integer(value)
    if not is_non_negative_integer(value) then
        return nil
    end

    local number = tonumber(value)

    if not number or number > MAX_SAFE_INTEGER then
        return nil
    end

    return number
end

-- 1. Validate arguments.
if not counter_key or counter_key == ""
    or not marker_key or marker_key == "" then
    return {-1, "invalid_arguments"}
end

local reservation_num = parse_integer(reservation)
local limit_num = parse_integer(limit)

if not reservation_num or reservation_num <= 0
    or not limit_num or limit_num <= 0 then
    return {-1, "invalid_arguments"}
end

-- 2. Check the idempotency marker before inspecting the counter.
local existing_marker = redis.call("GET", marker_key)

if existing_marker then
    local existing_amount = parse_integer(existing_marker)

    if not existing_amount or existing_amount ~= reservation_num then
        return {4, "corrupt"}
    end

    return {2, "already_reserved"}
end

-- 3. A missing counter requires recovery. Do not write anything.
local current_raw = redis.call("GET", counter_key)

if not current_raw then
    return {3, "recovery_required"}
end

-- 4. Reject corrupt counters.
local current = parse_integer(current_raw)

if not current then
    return {4, "corrupt"}
end

-- 5. Reject reservations larger than the entire budget.
if reservation_num > limit_num then
    return {
        5,
        "oversized",
        current,
        limit_num,
        math.max(0, limit_num - current)
    }
end

-- Reject if the current counter already exceeds the limit,
-- or if this reservation would exceed the remaining budget.
if current > limit_num
    or reservation_num > limit_num - current then
    return {
        6,
        "exceeded",
        current,
        limit_num,
        math.max(0, limit_num - current)
    }
end

-- 6. Use the counter's remaining TTL for the marker.
local pttl = redis.call("PTTL", counter_key)

if pttl == -1 then
    return {7, "no_ttl"}
end

if pttl <= 0 then
    return {3, "recovery_required"}
end

-- 7. Update the counter and create the marker.
-- Redis executes the script atomically relative to other commands.
local new_total = redis.call("INCRBY", counter_key, reservation)
redis.call("SET", marker_key, reservation, "PX", pttl)

return {
    1,
    "accepted",
    new_total,
    limit_num,
    limit_num - new_total
}
