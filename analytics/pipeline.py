from ingest import read_frames
from detect import run_detection
from track import make_state, update_tracks
from rules import no_hardhat, no_vest, forklift


def run_pipeline(source, model, cfg):
    tracker_state = make_state()

    rule_states = {
        "no_hardhat": {},
        "no_vest": {},
        "forklift": {},
    }

    for frame_data in read_frames(source):
        timestamp_ms = frame_data["timestamp_ms"]
        detections = run_detection(model, frame_data, cfg)
        tracks = update_tracks(tracker_state, detections, timestamp_ms, cfg)

        violations = []
        violations += no_hardhat.check(tracks, timestamp_ms, rule_states["no_hardhat"], cfg)
        violations += no_vest.check(tracks, timestamp_ms, rule_states["no_vest"], cfg)
        violations += forklift.check(tracks, timestamp_ms, rule_states["forklift"], cfg)

        for v in violations:
            yield v
