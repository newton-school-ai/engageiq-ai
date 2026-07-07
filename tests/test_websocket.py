"""Tests for WebSocket endpoint."""

import base64
import json

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_websocket_connection():
    """Test WebSocket connection is established."""
    with client.websocket_connect("/ws/session/1") as websocket:
        assert websocket is not None


def test_websocket_frame_processing():
    """Test WebSocket processes frame and returns engagement score."""
    with client.websocket_connect("/ws/session/1") as websocket:
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_b64 = base64.b64encode(frame.tobytes()).decode()

        # Send frame
        websocket.send_text(json.dumps({"frame": frame_b64, "timestamp": 1.0}))

        # Receive response
        response = json.loads(websocket.receive_text())
        assert response["session_id"] == 1
        assert "engagement_score" in response
        assert response["status"] == "processed"


def test_websocket_disconnect():
    """Test WebSocket handles disconnect gracefully."""
    with client.websocket_connect("/ws/session/1") as websocket:
        websocket.close()
