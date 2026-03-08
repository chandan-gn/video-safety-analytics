from models import make_violation, make_violation_state
from track import compute_iou

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
        condition = not has_hardhat  # condition True means violation present

        violation = apply_hysteresis(
            state[track_id], condition, timestamp_ms,
            enter_duration_ms, exit_duration_ms,
            track_id, person["bbox"], person["confidence"],
            person["last_seen_frame"], "no_hardhat"
        )
        if violation:
            violations.append(violation)

    return violations


def apply_hysteresis(vstate, condition, timestamp_ms, enter_ms, exit_ms,
                     track_id, bbox, confidence, frame_id, violation_type):
    s = vstate["state"]

    if s == "inactive":
        if condition:
            vstate["state"] = "pending"
            vstate["condition_true_since_ms"] = timestamp_ms
            vstate["condition_false_since_ms"] = None

    elif s == "pending":
        if condition:
            if timestamp_ms - vstate["condition_true_since_ms"] >= enter_ms:
                vstate["state"] = "active"
                return make_violation(track_id, violation_type, bbox, confidence, frame_id, timestamp_ms)
        else:
            vstate["state"] = "inactive"
            vstate["condition_true_since_ms"] = None

    elif s == "active":
        if not condition:
            vstate["state"] = "grace"
            vstate["condition_false_since_ms"] = timestamp_ms
        else:
            return make_violation(track_id, violation_type, bbox, confidence, frame_id, timestamp_ms)

    elif s == "grace":
        if condition:
            vstate["state"] = "active"
            vstate["condition_false_since_ms"] = None
            return make_violation(track_id, violation_type, bbox, confidence, frame_id, timestamp_ms)
        else:
            if timestamp_ms - vstate["condition_false_since_ms"] >= exit_ms:
                vstate["state"] = "inactive"
                vstate["condition_false_since_ms"] = None

    return None
