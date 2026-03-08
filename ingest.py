import json
import cv2


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_frames(source):
    if source.endswith(".jsonl"):
        yield from read_jsonl(source)
    elif source.lower().endswith(IMAGE_EXTENSIONS):
        yield from read_image(source)
    else:
        yield from read_video(source)


def read_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")

    yield {
        "frame_id": 0,
        "frame": frame,
        "timestamp_ms": 0.0,
        "detections": None,
    }


def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = (frame_id / fps) * 1000

        yield {
            "frame_id": frame_id,
            "frame": frame,
            "timestamp_ms": timestamp_ms,
            "detections": None,
        }

        frame_id += 1

    cap.release()


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            yield {
                "frame_id": data["frame_id"],
                "frame": None,
                "timestamp_ms": data["timestamp_ms"],
                "detections": data["detections"],
            }
