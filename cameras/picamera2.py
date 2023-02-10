import numpy as np
from cameras.basecamera import BaseCamera
from typing import Union
# camera functions
from picamera2 import Picamera2
from picamera2.encoders import Encoder


class PiCamera2(BaseCamera):
    def __init__(self, framesize_raw: list, framesize_preview: list, **kwargs):
        super().__init__(framesize_raw, framesize_preview, **kwargs)

        self.init_camera()

    def __call__(self):
        if self.framesize_preview is None:
            return np.zeros((self.framesize_raw))

        return np.zeros((self.framesize_raw)), np.zeros((self.framesize_preview))

    def init_camera(self):
        # init camera device
        try:
            self.picamera_context = Picamera2()
            self.video_config = self.picamera_context.create_video_configuration(
                main={
                    "size": self.framesize_preview, "format": "RGB888"},
                raw={"format": "SRGGB12", 'size': self.framesize_preview}
            )
            self.picamera_context.configure(self.video_config)
            self.picamera_context.encode_stream_name = "raw"
            self.encoder = Encoder()

            self.set_controls()
        except:
            self.picamera_context = None

    def set_controls(self, gain=None, exposuretime=None, fps=None):
        super().set_controls(gain, exposuretime, fps)

        config = {
            "AnalogueGain": self.gain,
            "ExposureTime": self.exposuretime,
            "FrameRate": self.fps
        }

        self.picamera_context.set_controls(config)

    def start(self):
        super().start()

        # TODO: 0.4 seconds delay needed after set_controls?
        self.picamera_context.start()

        self.camera_started = True

    def stop(self):
        super().stop()

        try:
            self.picamera_context.stop()
        except Exception as e:
            print(e)

        # suspect camera is stopped
        self.camera_started = False

    def __call__(self) -> Union[list, dict]:
        if self.camera_started != True:
            raise Exception("Camera must be started first")
        return self.picamera_context.capture_buffers(["raw", "main"])
