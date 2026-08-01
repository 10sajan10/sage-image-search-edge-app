from image_search.config import IngestConfig, SearchConfig, redacted_url


def test_redacted_url_hides_camera_credentials() -> None:
    value = redacted_url("rtsp://camera-user:camera-password@192.0.2.10:10001/main")

    assert value == "rtsp://***:***@192.0.2.10:10001/main"
    assert "camera-user" not in value
    assert "camera-password" not in value


def test_redacted_url_preserves_camera_alias() -> None:
    assert redacted_url("bottom-camera") == "bottom-camera"


def test_oneshot_mode_captures_once_and_drains(monkeypatch) -> None:
    monkeypatch.setenv("RUN_MODE", "oneshot")
    monkeypatch.setenv("CAPTURE_SOURCE", "camera")
    monkeypatch.setenv("CAMERA", "bottom-camera")
    monkeypatch.setenv("MAX_CAPTURES", "0")
    monkeypatch.setenv("EXIT_WHEN_DRAINED", "false")

    config = IngestConfig.from_env()

    assert config.run_mode == "oneshot"
    assert config.max_captures == 1
    assert config.exit_when_drained is True


def test_invalid_run_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("RUN_MODE", "cron")

    try:
        IngestConfig.from_env()
    except ValueError as error:
        assert "RUN_MODE" in str(error)
    else:
        raise AssertionError("invalid RUN_MODE was accepted")


def test_capture_interval_default_matches_documented_value(monkeypatch) -> None:
    monkeypatch.setenv("CAPTURE_SOURCE", "directory")

    assert IngestConfig.from_env().capture_interval_seconds == 180


def test_default_top_k_cannot_exceed_maximum(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_TOP_K", "201")
    monkeypatch.setenv("MAX_TOP_K", "200")

    try:
        SearchConfig.from_env()
    except ValueError as error:
        assert "DEFAULT_TOP_K" in str(error)
    else:
        raise AssertionError("DEFAULT_TOP_K greater than MAX_TOP_K was accepted")


def test_heartbeat_interval_stays_below_healthcheck_max_age(monkeypatch) -> None:
    monkeypatch.setenv("CAPTURE_SOURCE", "directory")

    # healthcheck.py fails the liveness probe at HEARTBEAT_MAX_AGE_SECONDS=30.
    assert IngestConfig.from_env().heartbeat_interval_seconds < 30
