LEGAL_TRANSITIONS = {
    ("pending", "reserved"),
    ("pending", "released"),
    ("reserved", "settled"),
    ("reserved", "released"),
    ("reserved", "uncertain"),
    ("uncertain", "settled"),
    ("uncertain", "released"),
}


def can_transition(
    current_state: str,
    target_state: str,
) -> bool:
    return (current_state, target_state) in LEGAL_TRANSITIONS