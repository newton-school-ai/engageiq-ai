"""
WebSocket endpoint for real-time frame streaming.
Receives base64-encoded frames from browser, feeds to CV pipeline,
returns engagement scores back to client.
"""

import base64
import json
import logging
from typing import Dict

import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.ingestion.frame_extractor import FramePreprocessor

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages multiple concurrent WebSocket connections (one per student)."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(
            f"Session {session_id} connected. Total: {len(self.active_connections)}"
        )

    def disconnect(self, session_id: str) -> None:
        self.active_connections.pop(session_id, None)
        logger.info(
            f"Session {session_id} disconnected. Total: {len(self.active_connections)}"
        )

    async def send_json(self, session_id: str, data: dict) -> None:
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_text(json.dumps(data))


manager = ConnectionManager()
preprocessor = FramePreprocessor()


def decode_frame(frame_b64: str) -> np.ndarray:
    """Decode base64-encoded frame into OpenCV BGR image."""
    raw = base64.b64decode(frame_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((480, 640, 3))
    return frame


@router.websocket("/ws/session/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(default=""),
):
    await manager.connect(session_id, websocket)
    try:
        while True:
            raw_message = await websocket.receive_text()
            data = json.loads(raw_message)

            frame_b64 = data.get("frame", "")
            timestamp = data.get("timestamp", 0.0)

            if not frame_b64:
                await manager.send_json(
                    session_id,
                    {
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "error": "Missing frame data",
                        "status": "error",
                    },
                )
                continue

            frame = decode_frame(frame_b64)
            processed = preprocessor.preprocess(frame)
            engagement_score = float(processed.get("engagement_score", 0.75))

            await manager.send_json(
                session_id,
                {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "engagement_score": engagement_score,
                    "status": "processed",
                    "frame_shape": list(frame.shape),
                },
            )

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"Session {session_id} cleanly disconnected.")
    except Exception as e:
        logger.error(f"Session {session_id} error: {e}")
        manager.disconnect(session_id)
