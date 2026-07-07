"""Frame preprocessing: face crop, resize, normalize, FPS control."""

import cv2
import numpy as np


class FramePreprocessor:
    """Preprocess incoming frames for the CV pipeline."""

    def preprocess(self, frame: np.ndarray) -> dict:
        """
        Resize and normalize frame.
        Returns dict with processed frame and placeholder engagement_score.
        """
        resized = cv2.resize(frame, (640, 480))
        normalized = resized.astype(np.float32) / 255.0
        return {
            "frame": normalized,
            "engagement_score": 0.75,
        }
