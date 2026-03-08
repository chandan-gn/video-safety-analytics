from __future__ import annotations                                                                                                          

def make_detection(frame_id, class_id, class_name, bbox, confidence, centroid):
    return {
        "frame_id": frame_id,
        "class_id": class_id,
        "class_name": class_name,
        "bbox": bbox,               # (x1, y1, x2, y2)
        "confidence": confidence,
        "centroid": centroid,       # (cx, cy)
    }


def make_track(track_id, class_id, class_name, bbox, confidence, frame_id, timestamp_ms):
    return {
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "bbox": bbox,                           # smoothed each frame
        "confidence": confidence,               # smoothed each frame
        "frames_seen": 1,
        "frames_lost": 0,
        "first_seen_frame": frame_id,
        "last_seen_frame": frame_id,
        "first_seen_ms": timestamp_ms,
        "last_seen_ms": timestamp_ms,
    }


def make_violation(track_id, violation_type, bbox, confidence, frame_id, timestamp_ms):
    return {
        "track_id": track_id,
        "violation_type": violation_type,
        "bbox": bbox,
        "confidence": confidence,
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
    }


def make_violation_state():
    return {
        "state": "inactive",                    # inactive | pending | active | grace
        "condition_true_since_ms": None,
        "condition_false_since_ms": None,
    }
