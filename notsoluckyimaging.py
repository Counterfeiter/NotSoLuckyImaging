#!/usr/bin/python3
import time, os

import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import Encoder
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
from PIL import Image
from scipy.ndimage import laplace
from skimage.util.shape import view_as_windows
from skimage.measure import blur_effect
import skimage

import matplotlib.pyplot as plt

session = {
    "object_name": "M13",
    "dng_folder": "/home/pi/luckyimaging/",
    "tmp_folder": "/home/pi/ramdisk/"
}

raw_camera_settings = {
    "size": (1920, 1080),
    "bayerformat": "SGBRG12",
    "bit_depth": 12,
    "max_fps": 60,
    "analoggain": (1, 31) # (min, max)
}

manual_quality_settings = {
    "analyse_frames_per_settings": 5,
    "min_dynamic_range": 25, # %
    "percentile": (0.1, 99.9),
    #higher is better, no maximum
    "laplace_threshold": 150,
    # lower is better, range from 0 to 1.0
    "blur_threshold": 0.35,
    "use_gains": [1, 30],
    "use_exposures": [10e3, 200e3, 600e3, 1200e3]
}

picam2 = Picamera2()
video_config = picam2.create_video_configuration(raw={"format": raw_camera_settings["bayerformat"], 'size': raw_camera_settings["size"]})
picam2.configure(video_config)
picam2.encode_stream_name = "raw"
encoder = Encoder()

def stride_conv_strided(arr):

    # convolution to debayer to grayscale with 1/4 resolution
    arr2 = np.array([[0.2989, 0.5870/2],
                    [0.5870/2, 0.1140]])

    s = 2

    def strided4D(arr, arr2, s):
        return view_as_windows(arr, arr2.shape, step=s)

    arr4D = strided4D(arr,arr2,s=s)
    return np.tensordot(arr4D, arr2, axes=((2,3),(0,1)))


config = {
    "AeEnable": False, 
    "AwbEnable": False, 
    "FrameRate": raw_camera_settings["max_fps"],
    "ExposureTime": int(1e6 // raw_camera_settings["max_fps"]),
    "AnalogueGain": 1.0,
    #"ScalerCrop": offset + size # roi could be used later
}


results = []

max_adc_range = 2**raw_camera_settings["bit_depth"] - 1
pixel_cnt = raw_camera_settings["size"][0] * raw_camera_settings["size"][1]

for gain in manual_quality_settings["use_gains"]:
    for exposure in manual_quality_settings["use_exposures"]:

        # fps req. for the given expsoure
        fps_exposure = 1e6 / exposure

        # do camera settings
        config["AnalogueGain"] = gain
        config["ExposureTime"] = int(exposure)
        config["FrameRate"] = min(fps_exposure, 30)
        picam2.set_controls(config)

        # record into a ramfs or tmpfs storage for speed
        tmp_video_path = os.path.join(session["tmp_folder"], "tmp.raw")
        tmp_pts_path = os.path.join(session["tmp_folder"], "timestamp.txt")
        picam2.start_recording(encoder, tmp_video_path, pts=tmp_pts_path)
        # for small exposures record min two seconds to grab enouth frames
        time.sleep(max(2, manual_quality_settings["analyse_frames_per_settings"] * 1/fps_exposure))
        picam2.stop_recording()

        framelist = []

        # read requested frames in the video
        try:
            with open(tmp_video_path, "rb") as file_p:
                for _ in range(manual_quality_settings["analyse_frames_per_settings"] + 1):
                    buf = file_p.read(pixel_cnt * 2)
                    framelist.append(np.frombuffer(buf, dtype=np.uint16).reshape((raw_camera_settings["size"][1], raw_camera_settings["size"][0])))
        except Exception as e:
            print(e)

        print("{:d} frames read - start processing".format(len(framelist)))

        for i, arr in enumerate(framelist):
            # discard the first frame - possible other settings
            if i == 0: 
                continue
            
            if i > manual_quality_settings["analyse_frames_per_settings"]:
                break

            min_value, max_value = np.percentile(arr, manual_quality_settings["percentile"])

            range_percent = (max_value - min_value) / max_adc_range * 100

            if range_percent < manual_quality_settings["min_dynamic_range"]:
                print("Low dynamic range (min {:.1f}, max {:.1f}) detected - skip". format(min_value, max_value))
                continue

            if max_value >= max_adc_range:
                print("Sensor saturation - skip")
                continue
            
            # convert bayer to grayscale with convolution - there is no fancy debayer algo. ongoing
            new_array = stride_conv_strided(arr)

            # calulate histogram
            hist, _ = np.histogram(new_array, 64, (0, max_adc_range))

            # first blure test option - laplace kernel
            laplaceout = laplace(new_array)

            # second blure test option - skikit image
            blur_metric = blur_effect(new_array)

            # stretch the grayscale image to display it later
            new_array = skimage.exposure.rescale_intensity( new_array, out_range=(100, max_adc_range-100))


            out = {
                "config" : config,
                "min" : min_value,
                "max" : max_value,
                "gray" : new_array,
                "histogram" : hist,
                "raw" : arr,
                "laplace" : np.std(laplaceout),
                "blurmetric" : blur_metric
            }

            results.append(out)

            print("Min: {:.0f} Max: {:.0f}, Laplace: {:.1f} Blur: {:.3f}".format(out["min"], out["max"] , out["laplace"], out["blurmetric"]))
        

# take the worst and the best to display the gray scale images
bestseeing = min(results, key=lambda x:x['blurmetric'])
worstseeing = max(results, key=lambda x:x['blurmetric'])

### display grayscale images
f, axarr = plt.subplots(2,2)
image_to_show = bestseeing["gray"]
print("Best blure", bestseeing['blurmetric'])
axarr[0,0].imshow(image_to_show, cmap='gray', vmin=0, vmax=max_adc_range)
image_to_show = worstseeing["gray"]
print("Worst blure", worstseeing['blurmetric'])
axarr[1,0].imshow(image_to_show, cmap='gray', vmin=0, vmax=max_adc_range)

### display historgrams
x_histo = np.linspace(0, max_adc_range, len(bestseeing["histogram"]), endpoint = False)
axarr[0,1].plot(x_histo, bestseeing["histogram"])
axarr[0,1].set_yscale("log")
axarr[1,1].plot(x_histo, worstseeing["histogram"])
axarr[1,1].set_yscale("log")



# debug output - save all gray scale images as png and name it with the blure metric
for res in results:
    plt.imsave('/home/pi/ramdisk/pngs/{:.3f}.png'.format(res["blurmetric"]), res["gray"], cmap='gray', vmin=0, vmax=max_adc_range)


    #store only selected raw images
    if res["laplace"] < manual_quality_settings["laplace_threshold"]:
        continue

    if res["blurmetric"] > manual_quality_settings["blur_threshold"]:
        continue

    # we take this from capture_video_raw example
    # Create DNG file from frame, based on https://github.com/schoolpost/PiDNG/blob/master/examples/raw2dng.py
    r = RAW2DNG()
    t = DNGTags()
    t.set(Tag.ImageWidth, raw_camera_settings["size"][0])
    t.set(Tag.ImageLength, raw_camera_settings["size"][1])
    t.set(Tag.TileWidth, raw_camera_settings["size"][0])
    t.set(Tag.TileLength, raw_camera_settings["size"][1])
    t.set(Tag.Orientation, Orientation.Horizontal)
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, raw_camera_settings["bit_depth"])
    t.set(Tag.CFARepeatPatternDim, [2, 2])
    t.set(Tag.CFAPattern, CFAPattern.RGGB)
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
    r.options(t, path="", compress=False)
    r.convert(arr, filename=os.path.join(session["dng_folder"], "{}_G{:02d}_E{:04d}".format(
        session["object_name"], res["config"]["AnalogueGain"], res["config"]["ExposureTime"] // 1000 )))


plt.show()