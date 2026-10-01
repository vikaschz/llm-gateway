local model = ARGV[1]
local window_id = ARGV[2]
local key_id = ARGV[3]
local delta = tonumber(ARGV[4])

local usage_key =
    "rotation:usage:" .. model .. ":" .. window_id

if redis.call(
    "EXISTS",
    usage_key
) == 0 then

    return {
        "EXPIRED"
    }

end

redis.call(
    "ZINCRBY",
    usage_key,
    delta,
    key_id
)

return {
    "UPDATED",
    delta
}