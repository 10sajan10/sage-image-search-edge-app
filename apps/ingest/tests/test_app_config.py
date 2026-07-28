from image_search.config import IngestConfig, redacted_url


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
