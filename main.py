import argparse
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="SafeStream Analytics Pipeline")
    parser.add_argument("--source", required=True, help="Path to .mp4, .jpg, or .jsonl file")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI server URL")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.source, "rb") as f:
        with httpx.stream("POST", f"{args.api_url}/run", files={"file": (os.path.basename(args.source), f)}, timeout=None) as response:
            for line in response.iter_lines():
                if line:
                    print(line)


if __name__ == "__main__":
    main()
