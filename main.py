import argparse
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from detect import load_model
from pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="SafeStream Analytics Pipeline")
    parser.add_argument("--source", required=True, help="Path to .mp4 or .jsonl file")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "pipeline.yaml"))
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI server URL")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = load_model(cfg["model"]["path"])

    for violation in run_pipeline(args.source, model, cfg):
        print(json.dumps(violation))
        try:
            httpx.post(f"{args.api_url}/violations", json=violation)
        except httpx.RequestError:
            pass  # API not running, continue processing


if __name__ == "__main__":
    main()
