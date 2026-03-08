from models import make_violation_state
from track import compute_iou
from rules.hysteresis import apply_hysteresis

ASSOCIATION_IOU = 0.15  # min IoU to associate a hard-hat with a person


def check(tracks, timestamp_ms, state, cfg):
    rule_cfg = cfg["rules"]["no_hardhat"]
    if not rule_cfg["enabled"]:
        return []

    min_confirmed_ms = cfg["tracker"]["min_confirmed_ms"]
    enter_duration_ms = rule_cfg["enter_duration_ms"]
    exit_duration_ms = rule_cfg["exit_duration_ms"]

    persons = [t for t in tracks if t["class_name"] == "person"]
    hardhats = [t for t in tracks if t["class_name"] == "hard-hat"]

    violations = []

    for person in persons:
        age_ms = timestamp_ms - person["first_seen_ms"]
        if age_ms < min_confirmed_ms:
            continue

        track_id = person["track_id"]
        if track_id not in state:
            state[track_id] = make_violation_state()

        has_hardhat = any(
            compute_iou(person["bbox"], hat["bbox"]) >= ASSOCIATION_IOU
            for hat in hardhats
        )
        condition = not has_hardhat

        violation = apply_hysteresis(
            state[track_id], condition, timestamp_ms,
            enter_duration_ms, exit_duration_ms,
            track_id, person["bbox"], person["confidence"],
            person["last_seen_frame"], "no_hardhat"
        )
        if violation:
            violations.append(violation)

    return violations
