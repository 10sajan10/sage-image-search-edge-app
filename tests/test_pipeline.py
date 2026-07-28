from __future__ import annotations

from pathlib import Path

import pytest

from app_config import CommonConfig, SearchConfig
from pipeline import SearchEngine, normalize


class FakeEmbedder:
    loaded = True
    device = "test"

    def encode_text(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeStore:
    def __init__(self):
        self.calls = []
        self.points = {
            "image": [
                {"id": "a", "score": 0.9, "payload": {"caption": "fire", "image_id": "a.jpg"}},
                {"id": "b", "score": 0.5, "payload": {"caption": "cloud", "image_id": "b.jpg"}},
            ],
            "caption": [
                {"id": "a", "score": 0.8, "payload": {"caption": "fire", "image_id": "a.jpg"}},
                {"id": "b", "score": 0.6, "payload": {"caption": "cloud", "image_id": "b.jpg"}},
            ],
            "caption_bm25": [
                {"id": "a", "score": 2.0, "payload": {"caption": "fire", "image_id": "a.jpg"}},
            ],
        }

    def query(self, using, query, limit, query_filter):
        self.calls.append((using, query, limit, query_filter))
        return self.points[using]


def search_config() -> SearchConfig:
    common = CommonConfig(
        qdrant_url="http://qdrant:6333",
        collection="test",
        vector_dim=3,
        bm25_model="Qdrant/bm25",
        embed_model_dir=Path("/models/test"),
        node_id="node",
        camera_id="camera",
        require_gpu=False,
        http_retries=0,
        http_backoff_seconds=0,
    )
    return SearchConfig(
        common=common,
        host="127.0.0.1",
        port=8099,
        default_weights={"image": 0.6, "caption": 0.25, "bm25": 0.15},
        default_top_k=10,
        max_top_k=100,
        candidates_per_leg=20,
        max_concurrency=1,
        request_timeout_seconds=10,
        api_key="",
    )


def test_normalize() -> None:
    assert normalize({}) == {}
    assert normalize({"a": 5.0}) == {"a": 1.0}
    assert normalize({"a": 2.0, "b": 4.0}) == {"a": 0.0, "b": 1.0}


def test_weighted_search_returns_explainable_scores() -> None:
    store = FakeStore()
    engine = SearchEngine(search_config(), store, FakeEmbedder())
    result = engine.search("wildfire", 2)

    assert result["legs_queried"] == ["bm25", "caption", "image"]
    assert result["results"][0]["id"] == "a"
    assert result["results"][0]["score"] == 1.0
    assert result["results"][0]["scores"] == {
        "image": 1.0,
        "caption": 1.0,
        "bm25": 1.0,
    }
    assert result["results"][0]["raw_scores"]["bm25"] == 2.0
    assert result["results"][0]["caption"] == "fire"


def test_zero_weight_skips_leg_and_filters_are_forwarded() -> None:
    store = FakeStore()
    engine = SearchEngine(search_config(), store, FakeEmbedder())
    result = engine.search(
        "fire",
        2,
        {"image": 0, "caption": 0, "bm25": 1},
        since=100,
        node_id="H01E",
    )

    assert result["legs_queried"] == ["bm25"]
    assert len(store.calls) == 1
    assert store.calls[0][0] == "caption_bm25"
    assert store.calls[0][3] == {
        "must": [
            {"key": "timestamp", "range": {"gte": 100}},
            {"key": "node_id", "match": {"value": "H01E"}},
        ]
    }


@pytest.mark.parametrize(
    "weights",
    [
        {"bogus": 1},
        {"image": -1},
        {"image": 0, "caption": 0, "bm25": 0},
        {"image": float("nan")},
    ],
)
def test_invalid_weights_are_rejected(weights) -> None:
    engine = SearchEngine(search_config(), FakeStore(), FakeEmbedder())
    with pytest.raises(ValueError):
        engine.resolve_weights(weights)
