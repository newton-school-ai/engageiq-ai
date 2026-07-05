"""Frame preprocessing: face crop, resize, normalize, FPS control."""

import cv2
import numpy as np


class FramePreprocessor:
    """
    Converts raw BGR webcam frames into ML-ready RGB frames:
    resize -> BGR to RGB -> normalize pixel values to [0, 1].
    """

    def __init__(self, target_size=(640, 480), normalize=True):
        self.target_size = target_size
        self.normalize = normalize

    def process(self, frame):
        if frame is None:
            raise ValueError("Frame cannot be None")

        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.normalize:
            rgb = rgb.astype(np.float32) / 255.0

        return rgb

    def batch_process(self, frames):
        return np.stack([self.process(f) for f in frames])
