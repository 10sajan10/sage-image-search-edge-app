from __future__ import annotations

from cli import print_results


def test_print_results_includes_accessible_image_path(capsys) -> None:
    print_results(
        {
            "query": "cloud",
            "returned": 1,
            "took_ms": 1.0,
            "weights_used": {"image": 0.6, "caption": 0.25, "bm25": 0.15},
            "results": [
                {
                    "id": "point-1",
                    "image_id": "camera/frame.jpg",
                    "image_path": "/data/frames/camera/frame.jpg",
                    "image_available": True,
                    "caption": "A cloud over the camera.",
                    "score": 1.0,
                    "scores": {"image": 1.0, "caption": 1.0, "bm25": 1.0},
                }
            ],
        }
    )

    output = capsys.readouterr().out
    assert "path: /data/frames/camera/frame.jpg (available)" in output
