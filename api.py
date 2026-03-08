import json
import os
import shutil
import sys
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from detect import load_model
from pipeline import run_pipeline

cfg = load_config(os.path.join(os.path.dirname(__file__), "pipeline.yaml"))
model = load_model(os.path.join(os.path.dirname(__file__), cfg["model"]["path"]))

app = FastAPI()

violations_store = {}  # key: (track_id, violation_type) → latest violation


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/violations")
def get_violations():
    return list(violations_store.values())


@app.post("/run")
def run(file: UploadFile = File(...)):
    name = file.filename.lower()
    if name.endswith(".jsonl"):
        suffix = ".jsonl"
    elif name.endswith((".jpg", ".jpeg")):
        suffix = ".jpg"
    elif name.endswith(".png"):
        suffix = ".png"
    elif name.endswith(".bmp"):
        suffix = ".bmp"
    elif name.endswith(".webp"):
        suffix = ".webp"
    else:
        suffix = ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    def stream():
        try:
            for violation in run_pipeline(tmp_path, model, cfg):
                key = (violation["track_id"], violation["violation_type"])
                violations_store[key] = violation
                yield json.dumps(violation) + "\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(stream(), media_type="application/x-ndjson")
