from models import make_violation


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
