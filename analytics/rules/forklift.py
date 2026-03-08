import math

from models import make_violation_state
from rules.hysteresis import apply_hysteresis


def check(tracks, timestamp_ms, state, cfg):
    rule_cfg = cfg["rules"]["forklift_proximity"]
    if not rule_cfg["enabled"]:
        return []

    min_confirmed_ms = cfg["tracker"]["min_confirmed_ms"]
    enter_duration_ms = rule_cfg["enter_duration_ms"]
    exit_duration_ms = rule_cfg["exit_duration_ms"]
    distance_threshold = rule_cfg["distance_threshold_px"]

    persons = [t for t in tracks if t["class_name"] == "person"]
    forklifts = [t for t in tracks if t["class_name"] == "forklift"]

    if not forklifts:
        return []

    violations = []

    for person in persons:
        age_ms = timestamp_ms - person["first_seen_ms"]
        if age_ms < min_confirmed_ms:
            continue

        track_id = person["track_id"]
        if track_id not in state:
            state[track_id] = make_violation_state()

        pcx, pcy = person["centroid"]
        too_close = any(
            math.dist((pcx, pcy), forklift["centroid"]) < distance_threshold
            for forklift in forklifts
        )

        violation = apply_hysteresis(
            state[track_id], too_close, timestamp_ms,
            enter_duration_ms, exit_duration_ms,
            track_id, person["bbox"], person["confidence"],
            person["last_seen_frame"], "forklift_proximity"
        )
        if violation:
            violations.append(violation)

    return violations
