"""OpenCV boundary. Native libraries are imported only when capture is requested."""
import os
from pathlib import Path


class OpenCVStream:
    def __init__(self, source, source_type, settings):
        # Suppress native decoder logs that can include credential-bearing URLs.
        os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
        os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
        import cv2
        if hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)
        self.cv2 = cv2
        if source_type == 'rtsp':
            if not source.lower().startswith(('rtsp://', 'rtsps://')):
                raise ValueError('Configured camera source is not RTSP')
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, settings.read_timeout_ms,
            ])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            if not Path(source).is_file():
                raise FileNotFoundError('Configured replay file is unavailable')
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise ConnectionError('Capture unavailable')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 0 < fps < 240 else 25.0
        self.frame_count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) if source_type == 'file' else 0

    def read(self):
        return self.cap.read()

    def rewind(self):
        if getattr(self, 'cap', None) and self.cap.isOpened():
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            return True
        return False

    def close(self):
        self.cap.release()
