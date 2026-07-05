"""Webcam capture pipeline supporting webcam, file, and RTSP inputs."""

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np


class WebcamCaptureError(Exception):
    """Raised when the capture source cannot be opened or read."""
    pass


@dataclass
class Frame:
    """A single captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int


class WebcamCapture:
    """
    Captures frames from a webcam or video file at a configurable FPS
    and resolution, attaching a timestamp to every frame.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        fps: int = 15,
        resolution: tuple = (1280, 720),
    ):
        self.source = source
        self.target_fps = fps
        self.resolution = resolution
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._frame_interval = 1.0 / fps

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise WebcamCaptureError(
                f"Could not open capture source '{self.source}'. "
                "Check that a webcam is connected or the file path is correct."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise WebcamCaptureError("Capture source not opened. Call open() first.")

        ret, image = self._cap.read()
        if not ret:
            return None

        frame = Frame(
            image=image,
            timestamp=time.time(),
            frame_number=self._frame_count,
        )
        self._frame_count += 1
        return frame

    def frames(self, duration: Optional[float] = None):
        start_time = time.time()
        next_frame_time = start_time

        while True:
            if duration is not None and (time.time() - start_time) >= duration:
                break

            frame = self.read()
            if frame is None:
                break

            yield frame

            next_frame_time += self._frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_actual_fps(self, sample_duration: float = 3.0) -> float:
        count = 0
        start = time.time()
        for _ in self.frames(duration=sample_duration):
            count += 1
        elapsed = time.time() - start
        return count / elapsed if elapsed > 0 else 0.0

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def main():
    parser = argparse.ArgumentParser(description="Test webcam capture")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS")
    parser.add_argument("--duration", type=float, default=5.0, help="Capture duration in seconds")
    parser.add_argument("--source", default=0, help="Webcam index or video file path")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source

    try:
        with WebcamCapture(source=source, fps=args.fps) as cap:
            print(f"Capturing for {args.duration}s at target {args.fps} FPS...")
            count = 0
            start = time.time()
            for frame in cap.frames(duration=args.duration):
                count += 1
            elapsed = time.time() - start
            actual_fps = count / elapsed if elapsed > 0 else 0
            print(f"Captured {count} frames in {elapsed:.2f}s -> actual FPS: {actual_fps:.2f}")
    except WebcamCaptureError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
