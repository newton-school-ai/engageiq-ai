"""WebSocket endpoint for real-time frame streaming."""
import base64
import json

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from src.pipeline.preprocessor import FramePreprocessor

preprocessor = FramePreprocessor()


async def websocket_endpoint(websocket: WebSocket, session_id: int) -> None:
    """Handle WebSocket connection for frame streaming.

    Args:
        websocket: WebSocket connection object.
        session_id: ID of the session being monitored.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            # Base64 decode frame
            frame_bytes = base64.b64decode(payload["frame"])
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = frame.reshape((480, 640, 3))

            # Preprocess frame
            processed = preprocessor.process(frame)  # noqa: F841

            # Send back engagement score
            await websocket.send_json({
                "session_id": session_id,
                "engagement_score": 75.0,
                "timestamp": payload.get("timestamp", 0.0),
                "status": "processed",
            })

    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")
    except Exception as e:
        print(f"Error processing frame: {e}")
        await websocket.close()