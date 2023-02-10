from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image
import numpy as np
from cameras.basecamera import BaseCamera
from typing import Union
import glob
import os
from astropy.io import fits
from skimage.util.shape import view_as_windows
from skimage.transform import resize


class FolderLoopCamera(BaseCamera):
    def __init__(self, framesize_raw: list, framesize_preview: list, **kwargs):
        super().__init__(framesize_raw, framesize_preview, **kwargs)

        self.scanfolder = kwargs["scanfolder"]
        self.init_camera()

    def init_camera(self):
        # init camera device
        path = os.path.join(self.scanfolder, "*.fit*")
        self.filelist = glob.glob(path)
        self.fileindex = 0

    @staticmethod
    def stride_conv_strided(arr):
        # convolution to debayer to grayscale with 1/4 resolution
        arr2 = np.array([[0.2989, 0.5870/2],
                        [0.5870/2, 0.1140]])

        s = 2

        def strided4D(arr, arr2, s):
            return view_as_windows(arr, arr2.shape, step=s)

        arr4D = strided4D(arr, arr2, s=s)
        return np.tensordot(arr4D, arr2, axes=((2, 3), (0, 1)))

    def set_controls(self, gain=None, exposuretime=None, fps=None):
        super().set_controls(gain, exposuretime, fps)

        pass

    def start(self):
        super().start()

        self.camera_started = True

    def stop(self):
        super().stop()

        self.camera_started = False

    def __call__(self) -> Union[list, dict]:
        # we use the simulated exposure and fps delay from the base class
        dummy_content = super().__call__()

        if len(self.filelist) <= self.fileindex:
            self.fileindex = 0
        else:
            self.fileindex += 1

        try:
            hdu_list = fits.open(self.filelist[self.fileindex])
            if hdu_list[0].data.ndim == 3:
                # take only the frist channel if its properly a color layered image
                data_load = hdu_list[0].data[0]
                data_preview = resize(
                    data_load, self.framesize_preview, anti_aliasing=True)
                data_load = resize(
                    data_load, self.framesize_raw, anti_aliasing=True)
            elif hdu_list[0].data.ndim == 2:
                data_load = hdu_list[0].data
                # debayer preview
                data_preview = FolderLoopCamera.stride_conv_strided(data_load)

                # resize
                data_preview = resize(
                    data_preview, self.framesize_preview[::-1], anti_aliasing=True)

                data_preview /= 4  # from 12 to 8 bit
                data_preview = np.clip(data_preview, 0, 255).astype(np.uint8)

                # NOTE: we could not resize a bayerd image - return as given
            else:
                print("Fits image has wrong dimension")
                return dummy_content
        except Exception as e:
            print(e)
            return dummy_content

        if self.framesize_preview is None:
            return [data_load.flatten()], {}
        return [data_load.flatten(), data_preview.flatten()], {}
