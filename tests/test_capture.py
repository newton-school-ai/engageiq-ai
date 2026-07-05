from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.webcam_capture import WebcamCapture, WebcamCaptureError


def test_capture_initialization():
    cap = WebcamCapture(source=0, fps=15, resolution=(1280, 720))
    assert cap.target_fps == 15
    assert cap.resolution == (1280, 720)
    assert cap._cap is None


@patch("cv2.VideoCapture")
def test_open_raises_on_missing_webcam(mock_video_capture):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = False
    mock_video_capture.return_value = mock_instance

    cap = WebcamCapture(source=0)
    with pytest.raises(WebcamCaptureError):
        cap.open()


@patch("cv2.VideoCapture")
def test_read_returns_frame_with_timestamp(mock_video_capture):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_video_capture.return_value = mock_instance

    cap = WebcamCapture(source=0)
    cap.open()
    frame = cap.read()

    assert frame is not None
    assert frame.image.shape == (720, 1280, 3)
    assert frame.timestamp > 0
    assert frame.frame_number == 0
