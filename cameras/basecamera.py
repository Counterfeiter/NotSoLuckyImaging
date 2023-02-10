import numpy as np
import time
from typing import Union


class BaseCamera:
    def __init__(self, framesize_raw: list, framesize_preview: list = None, **kwargs):
        if "gain" in kwargs:
            self.gain = kwargs["gain"]
        else:
            self.gain = 1

        if "exposuretime" in kwargs:
            self.exposuretime = kwargs["exposuretime"]
        else:
            self.exposuretime = 10000  # us

        if "fps" in kwargs:
            self.fps = 30
        else:
            self.fps = min([30.0, 1e6 / self.exposuretime])

        self.framesize_raw = framesize_raw
        self.framesize_preview = framesize_preview

        self.last_frame_timestamp = time.time()
        self.camera_started = False

    def __call__(self) -> Union[list, dict]:
        wait_time = max([1.0/self.fps, self.exposuretime / 1e6])
        sleep_time = max(
            [(wait_time - time.time() + self.last_frame_timestamp), 0])
        time.sleep(sleep_time)
        self.last_frame_timestamp = time.time()
        if self.framesize_preview is None:
            return [np.zeros(self.framesize_raw, dtype=np.uint16).flatten()], {}

        return [np.zeros(self.framesize_raw, dtype=np.uint16).flatten(), np.zeros(self.framesize_preview, dtype=np.uint8).flatten()], {}

    def set_controls(self, gain=None, exposuretime=None, fps=None):
        self.gain = self.gain if gain is None else gain
        self.exposuretime = self.exposuretime if exposuretime is None else exposuretime
        self.fps = self.fps if fps is None else fps

    def start(self):
        self.last_frame_timestamp = time.time()
        if self.camera_started:
            raise Exception("Camera is already started")

    def stop(self):
        if not self.camera_started:
            raise Exception("Camera is already stopped")
