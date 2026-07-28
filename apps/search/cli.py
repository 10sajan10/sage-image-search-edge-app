"""Terminal client for the deployed search API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language image query")
    parser.add_argument(
        "--url",
        default=os.environ.get("SEARCH_API_URL", "http://127.0.0.1:8099"),
        help="Search API base URL",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("SEARCH_API_KEY", "")
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--image-weight", type=float)
    parser.add_argument("--caption-weight", type=float)
    parser.add_argument("--bm25-weight", type=float)
    parser.add_argument("--since", type=int)
    parser.add_argument("--source")
    parser.add_argument("--node-id")
    parser.add_argument("--camera-id")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def request_payload(args: argparse.Namespace) -> dict:
    weights = {
        name: value
        for name, value in (
            ("image", args.image_weight),
            ("caption", args.caption_weight),
            ("bm25", args.bm25_weight),
        )
        if value is not None
    }
    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "since": args.since,
        "source": args.source,
        "node_id": args.node_id,
        "camera_id": args.camera_id,
    }
    if weights:
        payload["weights"] = weights
    return {key: value for key, value in payload.items() if value is not None}


def call_api(args: argparse.Namespace) -> dict:
    body = json.dumps(request_payload(args)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key.strip():
        headers["X-API-Key"] = args.api_key.strip()
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/search",
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"search API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"search API is unreachable: {error}") from error


def print_results(result: dict) -> None:
    weights = ", ".join(
        f"{key}={value:.3f}" for key, value in result["weights_used"].items()
    )
    print(
        f"query={result['query']!r} returned={result['returned']} "
        f"took={result['took_ms']}ms weights=[{weights}]"
    )
    for rank, item in enumerate(result["results"], start=1):
        leg_scores = " ".join(
            f"{key}={value:.3f}" for key, value in sorted(item["scores"].items())
        )
        caption = " ".join((item.get("caption") or "").split())
        if len(caption) > 180:
            caption = f"{caption[:177]}..."
        print(
            f"{rank:>2}. {item['score']:.4f} [{leg_scores}] "
            f"{item.get('image_id') or item['id']}"
        )
        if caption:
            print(f"    {caption}")


def main() -> int:
    args = parse_args()
    try:
        result = call_api(args)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print_results(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
