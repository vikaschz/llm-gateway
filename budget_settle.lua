-- Atomic budget settlement / release.
--
-- KEYS[1] = budget:committed:<period>
-- KEYS[2] = budget:marker:<period>:<reservation_id>
--
-- ARGV[1] = actual cost (non-negative integer)
--
-- Return codes:
--  1 = SETTLED
--  2 = MARKER_MISSING
--  3 = COUNTER_MISSING
--  4 = NEGATIVE
--  5 = CORRUPT
-- -1 = INVALID_ARGUMENTS

local counter_key = KEYS[1]
local marker_key = KEYS[2]
local actual_cost_raw = ARGV[1]

local MAX_SAFE_INTEGER = 9007199254740991

local function parse_non_negative_integer(value)
    if value == nil or string.match(value, "^%d+$") == nil then
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

local actual_cost = parse_non_negative_integer(actual_cost_raw)

if actual_cost == nil then
    return {-1, "invalid_arguments"}
end

-- 2. Marker must exist.
local marker_raw = redis.call("GET", marker_key)

if not marker_raw then
    return {2, "marker_missing"}
end

local reserved_amount = parse_non_negative_integer(marker_raw)

if reserved_amount == nil then
    return {5, "corrupt_marker"}
end

-- 3. Counter must exist.
local counter_raw = redis.call("GET", counter_key)

if not counter_raw then
    return {3, "counter_missing"}
end

local current_total = parse_non_negative_integer(counter_raw)

if current_total == nil then
    return {5, "corrupt_counter"}
end

-- 4. Calculate the settlement delta.
local delta = actual_cost - reserved_amount

-- Never allow the counter to become negative.
if delta < 0 and current_total < math.abs(delta) then
    return {
        4,
        "negative",
        current_total,
        delta,
        reserved_amount,
        actual_cost
    }
end

-- 5. Apply the delta and remove the marker.
local new_total = redis.call("INCRBY", counter_key, delta)
redis.call("DEL", marker_key)

return {
    1,
    "settled",
    new_total,
    delta
}