"""Tests for WebSocket frame streaming endpoint."""

import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def make_frame_b64(height=480, width=640) -> str:
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf.tobytes()).decode()


def test_websocket_connection(client):
    with client.websocket_connect("/ws/session/test-s1?token=test") as ws:
        ws.send_text(json.dumps({"frame": make_frame_b64(), "timestamp": 1.0}))
        response = json.loads(ws.receive_text())
        assert response["session_id"] == "test-s1"
        assert response["status"] == "processed"


def test_frame_processing_returns_score(client):
    with client.websocket_connect("/ws/session/test-s2?token=test") as ws:
        ws.send_text(json.dumps({"frame": make_frame_b64(), "timestamp": 2.5}))
        response = json.loads(ws.receive_text())
        assert "engagement_score" in response
        assert isinstance(response["engagement_score"], float)
        assert 0.0 <= response["engagement_score"] <= 1.0
        assert response["timestamp"] == 2.5
        assert "frame_shape" in response


def test_missing_frame_returns_error(client):
    with client.websocket_connect("/ws/session/test-s3?token=test") as ws:
        ws.send_text(json.dumps({"timestamp": 1.0}))
        response = json.loads(ws.receive_text())
        assert response["status"] == "error"
        assert "error" in response


def test_multiple_concurrent_sessions(client):
    with client.websocket_connect("/ws/session/s1?token=test") as ws1:
        with client.websocket_connect("/ws/session/s2?token=test") as ws2:
            frame = make_frame_b64()
            ws1.send_text(json.dumps({"frame": frame, "timestamp": 1.0}))
            ws2.send_text(json.dumps({"frame": frame, "timestamp": 2.0}))
            r1 = json.loads(ws1.receive_text())
            r2 = json.loads(ws2.receive_text())
            assert r1["session_id"] == "s1"
            assert r2["session_id"] == "s2"
