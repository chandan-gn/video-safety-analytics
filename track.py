import numpy as np
from scipy.optimize import linear_sum_assignment

from models import make_track


def make_state():
    return {
        "next_id": 0,
        "tracks": {},   # track_id -> track dict
    }


def compute_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def build_cost_matrix(tracks, detections):
    track_list = list(tracks.values())
    cost = np.ones((len(track_list), len(detections)))

    for i, track in enumerate(track_list):
        for j, det in enumerate(detections):
            if track["class_id"] == det["class_id"]:
                cost[i, j] = 1.0 - compute_iou(track["bbox"], det["bbox"])

    return cost, track_list


def update_tracks(state, detections, timestamp_ms, cfg):
    iou_threshold = cfg["tracker"]["iou_threshold"]
    alpha = cfg["tracker"]["ema_alpha"]
    max_frames_lost = cfg["tracker"]["max_frames_lost"]

    tracks = state["tracks"]

    if not tracks:
        for det in detections:
            track_id = state["next_id"]
            state["next_id"] += 1
            tracks[track_id] = make_track(
                track_id=track_id,
                class_id=det["class_id"],
                class_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                frame_id=det["frame_id"],
                timestamp_ms=timestamp_ms,
            )
        return list(tracks.values())

    cost, track_list = build_cost_matrix(tracks, detections)
    row_ids, col_ids = linear_sum_assignment(cost)

    matched_track_ids = set()
    matched_det_ids = set()

    for row, col in zip(row_ids, col_ids):
        if cost[row, col] > 1.0 - iou_threshold:
            continue

        track = track_list[row]
        det = detections[col]

        # EMA smoothing on bbox and confidence
        old_bbox = track["bbox"]
        new_bbox = det["bbox"]
        track["bbox"] = tuple(alpha * n + (1 - alpha) * o for n, o in zip(new_bbox, old_bbox))
        track["confidence"] = alpha * det["confidence"] + (1 - alpha) * track["confidence"]

        # update centroid from smoothed bbox
        x1, y1, x2, y2 = track["bbox"]
        track["centroid"] = ((x1 + x2) / 2, (y1 + y2) / 2)

        track["frames_seen"] += 1
        track["frames_lost"] = 0
        track["last_seen_frame"] = det["frame_id"]
        track["last_seen_ms"] = timestamp_ms

        matched_track_ids.add(track["track_id"])
        matched_det_ids.add(col)

    # unmatched detections → new tracks
    for j, det in enumerate(detections):
        if j not in matched_det_ids:
            track_id = state["next_id"]
            state["next_id"] += 1
            tracks[track_id] = make_track(
                track_id=track_id,
                class_id=det["class_id"],
                class_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                frame_id=det["frame_id"],
                timestamp_ms=timestamp_ms,
            )

    # unmatched tracks → increment lost, drop if exceeded
    for track_id in list(tracks.keys()):
        if track_id not in matched_track_ids:
            tracks[track_id]["frames_lost"] += 1
            if tracks[track_id]["frames_lost"] > max_frames_lost:
                del tracks[track_id]

    return list(tracks.values())
