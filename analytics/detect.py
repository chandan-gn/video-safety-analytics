import numpy as np
import onnxruntime as ort
import cv2

from models import make_detection

CLASS_MAP = {
    0: "person",
    1: "hard-hat",
    2: "gloves",
    3: "mask",
    4: "glasses",
    5: "boots",
    6: "vest",
    7: "ppe-suit",
    8: "ear-protector",
    9: "safety-harness",
    10: "no-hard-hat",
    11: "no-gloves",
    12: "no-mask",
    13: "no-glasses",
    14: "no-boots",
    15: "no-vest",
    16: "no-ppe-suit",
}

INPUT_SIZE = 320


def load_model(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session


def run_detection(session, frame_data, cfg):
    if frame_data["detections"] is not None:
        return frame_data["detections"]

    frame = frame_data["frame"]
    frame_id = frame_data["frame_id"]
    timestamp_ms = frame_data["timestamp_ms"]
    conf_threshold = cfg["model"]["confidence_threshold"]
    iou_threshold = cfg["model"]["iou_threshold"]

    input_tensor, scale, pad = preprocess(frame)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    detections = postprocess(outputs[0], scale, pad, frame_id, conf_threshold, iou_threshold)

    return detections


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    pad_w = (INPUT_SIZE - new_w) // 2
    pad_h = (INPUT_SIZE - new_h) // 2

    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 320, 320)

    return tensor, scale, (pad_w, pad_h)


def postprocess(output, scale, pad, frame_id, conf_threshold, iou_threshold):
    # output shape: (1, 21, 2100) → transpose to (2100, 21)
    predictions = output[0].T  # (2100, 21)

    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        cx, cy, bw, bh = pred[:4]
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < conf_threshold:
            continue

        pad_w, pad_h = pad
        x1 = (cx - bw / 2 - pad_w) / scale
        y1 = (cy - bh / 2 - pad_h) / scale
        x2 = (cx + bw / 2 - pad_w) / scale
        y2 = (cy + bh / 2 - pad_h) / scale

        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes],
        scores,
        conf_threshold,
        iou_threshold,
    )

    detections = []
    for i in indices:
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        class_id = class_ids[i]
        detections.append(make_detection(
            frame_id=frame_id,
            class_id=class_id,
            class_name=CLASS_MAP.get(class_id, "unknown"),
            bbox=(x1, y1, x2, y2),
            confidence=float(scores[i]),
            centroid=(cx, cy),
        ))

    return detections
