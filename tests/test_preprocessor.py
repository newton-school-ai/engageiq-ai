import numpy as np

from src.ingestion.frame_extractor import FramePreprocessor


def test_preprocessing_output_shape():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    prep = FramePreprocessor(target_size=(640, 480))
    result = prep.process(frame)

    assert result.shape == (480, 640, 3)


def test_preprocessing_normalizes_pixels():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    prep = FramePreprocessor(target_size=(640, 480), normalize=True)
    result = prep.process(frame)

    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_batch_process():
    frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    prep = FramePreprocessor(target_size=(640, 480))
    batch = prep.batch_process(frames)

    assert batch.shape == (3, 480, 640, 3)
