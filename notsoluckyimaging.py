#!/usr/bin/python3
import time, os, signal, sys, glob
import numpy as np
from datetime import datetime
from multiprocessing import Process, Pool, Queue
import queue # not process save
import matplotlib.pyplot as plt
import __main__ as main
import atexit
from collections import deque # not process save

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

#camera functions
from picamera2 import Picamera2
from picamera2.encoders import Encoder

### image functions
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
from PIL import Image
from scipy.ndimage import laplace
from skimage.util.shape import view_as_windows
from skimage.measure import blur_effect
from skimage.color import rgb2gray
import skimage
from astropy.io import fits

PRINT_TIME_PROFILING = False

session = {
    "object_name": "Jupiter",
    "save_image_dng" : False,
    "save_image_fits" : True,
    #"img_folder": "/home/pi/luckyimaging/",
    "img_folder" : "/media/pi/LOG_DONGLE/",
    "num_cpus": 4,
    "images_per_process": 1
}

raw_camera_settings = {
    "size": (1920, 1080),
    "size_processing": (800, 600),
    "bayerformat": "SRGGB12",
    "bit_depth": 12,
    "max_fps": 30, #max 60 but short exposure videos get very fast large (~4 MB per Frame)
    "analoggain": (1, 31) # (min, max)
}

manual_quality_settings = {
    "store_procent_of_passed_images": 1,
    "analyse_frames_per_settings": 20,
    "min_dynamic_range": 1, # %
    # allow three pixels (dead pixels?) to be saturated
    "percentile": (0.1, 100.0 - 3.0 / (raw_camera_settings["size"][0] * raw_camera_settings["size"][1]) * 100.0),
    #higher is better, no maximum
    "laplace_threshold": 1.0,
    # lower is better, range from 0 to 1.0
    "blur_threshold": None,
    "starfinder_threshold": None,
    "use_gains": [14], # unity gain at about 11, use > 14 to have lower read noise
    "use_exposures": [10e3] * 1
}

config = {
    "AeEnable": False, 
    "AwbEnable": False, 
    "FrameRate": raw_camera_settings["max_fps"],
    "ExposureTime": int(1e6 // raw_camera_settings["max_fps"]),
    "AnalogueGain": 1.0,
    #"ScalerCrop": offset + size # roi could be used later
}

max_adc_range = 2**raw_camera_settings["bit_depth"] - 1
pixel_cnt = raw_camera_settings["size"][0] * raw_camera_settings["size"][1]

picamera_setsaveexit = None

def process_frames(video_queue, results):

    while True:
        #st = datetime.timestamp(datetime.now())
        for buf, gray, threshold, metric in video_queue.get():
            #if PRINT_TIME_PROFILING:
            #    print("video_queue.get", datetime.timestamp(datetime.now()) - st)

            if buf is None:
                print("Close thread - work done")
                sys.exit(0)

            try:
                arr = np.frombuffer(buf, dtype=np.uint16).reshape((raw_camera_settings["size"][1], raw_camera_settings["size"][0]))
                arr_rgb = np.frombuffer(gray, dtype=np.uint8).reshape((raw_camera_settings["size_processing"][1], raw_camera_settings["size_processing"][0], 3))
            except Exception as e:
                print(e)
                continue

            st = datetime.timestamp(datetime.now())
            min_value, max_value = np.percentile(arr, manual_quality_settings["percentile"])
            if PRINT_TIME_PROFILING:
                print("percentile", datetime.timestamp(datetime.now()) - st)

            range_percent = (max_value - min_value) / max_adc_range * 100

            if max_value >= max_adc_range:
                print("Sensor saturation - skip")
                continue

            st = datetime.timestamp(datetime.now())
            # convert bayer to grayscale with convolution - there is no fancy debayer algo. ongoing
            gray_image = rgb2gray(arr_rgb.astype(np.float32))#stride_conv_strided(arr)
            if PRINT_TIME_PROFILING:
                print("rgb2gray", datetime.timestamp(datetime.now()) - st)

            image_passed = True
            laplaceout = None
            blur_metric = None
            star_quality = 0.0

            # first blure test option - laplace kernel
            if manual_quality_settings["laplace_threshold"] is not None:
                st = datetime.timestamp(datetime.now())
                laplaceout = np.std(laplace(gray_image))
                if laplaceout < manual_quality_settings["laplace_threshold"]:
                    image_passed = False

            # second blure test option - skikit image
            if manual_quality_settings["blur_threshold"] is not None:
                st = datetime.timestamp(datetime.now())
                blur_metric = blur_effect(gray_image)
                if blur_metric > manual_quality_settings["blur_threshold"]:
                    image_passed = False
                if PRINT_TIME_PROFILING:
                    print("blur_effect", datetime.timestamp(datetime.now()) - st)

            if manual_quality_settings["starfinder_threshold"] is not None:
                st = datetime.timestamp(datetime.now())
                mean, median, std = sigma_clipped_stats(gray_image, sigma=3.0)
                daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std, exclude_border=True, brightest=10, 
                                        peakmax=2**raw_camera_settings["bit_depth"]-1)
                try:
                    sources = daofind(gray_image - median)
                except:
                    pass # error if no stars are found
                else:
                    if sources is not None:
                        mean = 0.0
                        for src in sources:
                            mean += src["sharpness"]

                        if len(sources) > 0:
                            star_quality = mean / len(sources)

                if PRINT_TIME_PROFILING:
                    print("DAOStarFinder", datetime.timestamp(datetime.now()) - st)

            star_string = "{:.3f}".format(star_quality)
            blur_str = "{:.3f}".format(blur_metric) if blur_metric is not None else "None"
            laplace_str = "{:.1f}".format(laplaceout) if laplaceout is not None else "None"

            print("Min: {:.0f} Max: {:.0f}, Laplace: {} Blur: {} Starsharpness: {} {}".format(min_value, max_value , 
                        laplace_str, blur_str, star_string, "*" if image_passed else ""))

            out = {
                "config" : config,
                "min" : min_value,
                "max" : max_value,
                "laplace" : laplaceout,
                "blurmetric" : blur_metric,
                "starquality" : star_quality,
                "pass" : image_passed,
                "ts" : datetime.now()
            }
            # send calulated results to main thread
            results.put(out)

            if image_passed:
                print(out[metric], " <=> ", threshold)

                # check if given function marks image as storeable
                if metric == "laplace" or metric == "starquality":
                    if out[metric] <= threshold:
                        image_passed = False
                elif metric == "blurmetric":
                    if out[metric] >= threshold:
                        image_passed = False
                else:
                    image_passed = False
                
                if image_passed:
                    out["gray"] = gray_image
                    out["raw"] = arr
                    storeImages(out, metric)

def storeImages(res, metric):
    st = datetime.timestamp(datetime.now())

    if session["save_image_dng"]:
        writeDNGFile(res, metric)
    if session["save_image_fits"]:
        writeFITSFile(res, metric)

    if PRINT_TIME_PROFILING:
        print("store raw files", datetime.timestamp(datetime.now()) - st)

    return 0


def writeDNGFile(res_dict, metric):
    filename_dng = "{:.3f}{}_{}_G{:02d}_E{:04d}_{:f}".format(
        res_dict[metric], metric[0], session["object_name"], int(res_dict["config"]["AnalogueGain"]), 
        int(res_dict["config"]["ExposureTime"] // 1000), datetime.timestamp(res_dict["ts"]))
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
    t.set(Tag.Model, "IMX290C")
    r.options(t, path="", compress=False)
    r.convert(res_dict["raw"], filename=os.path.join(session["img_folder"], filename_dng ) )
    
    print(filename_dng + " saved!")

def writeFITSFile(res_dict, metric):
    filename_fits = "{:.3f}{}_{}_G{:02d}_E{:04d}_{:f}.fits".format(
        res_dict[metric], metric[0], session["object_name"], int(res_dict["config"]["AnalogueGain"]), 
        int(res_dict["config"]["ExposureTime"] // 1000), datetime.timestamp(res_dict["ts"]))

    hdu = fits.PrimaryHDU(data=res_dict["raw"])
    hdu.header["BSCALE"] = 1
    hdu.header["BITPIX"] = raw_camera_settings["bit_depth"]
    hdu.header["BZERO"] = 2**raw_camera_settings["bit_depth"]
    hdu.header["DATE-OBS"] = str(res_dict["ts"].isoformat())
    hdu.header["OBJECT"] = session["object_name"]
    hdu.header["OBSERVER"] = main.__file__
    hdu.header["IMAGETYP"] = "LIGHT"
    hdu.header["EXPOSURE"] = res_dict["config"]["ExposureTime"] // 1e6
    hdu.header["EXPTIME"] = hdu.header["EXPOSURE"]
    hdu.header["INSTRUME"] = "IMX290C"
    hdu.header["GAIN"] = res_dict["config"]["AnalogueGain"]
    hdu.header["XPIXSZ"] = raw_camera_settings["size"][0]
    hdu.header["YPIXSZ"] = raw_camera_settings["size"][1]
    hdu.header["BAYERPAT"] = "RGGB"
    hdu.header["XBAYROFF"] = 0
    hdu.header["YBAYROFF"] = 0
    #print(hdu.header)
    hdu.writeto(os.path.join(session["img_folder"], filename_fits ))

    print(filename_fits + " saved!")

def ctlC_handler(signum, frame):
    global picamera_setsaveexit
    print("Exit by Crtl + C requested... ")
    picamera_setsaveexit = True

if __name__ == "__main__":
    picamera_setsaveexit = False
    signal.signal(signal.SIGINT, ctlC_handler)

    # create a thread save list for the results
    results_queue = Queue()
    video_queue = Queue(session["num_cpus"])
    # video_path_queue = Queue(3)
    blur_quality_list = deque([0.0], maxlen=1000)
    images_processed = 0

    # init camera device
    picamera_context = Picamera2()
    video_config = picamera_context.create_video_configuration(
                            main={"size": raw_camera_settings["size_processing"], "format": "RGB888"},
                            raw={"format": raw_camera_settings["bayerformat"], 'size': raw_camera_settings["size"]})
    picamera_context.configure(video_config)
    picamera_context.encode_stream_name = "raw"
    encoder = Encoder()
    

    procs = []
    for _ in range(session["num_cpus"]):
        procs.append(Process(target=process_frames, args=(video_queue, results_queue)))
        procs[-1].start()

    start_time = datetime.timestamp(datetime.now())

    buffer_img = []

    persentile_side = manual_quality_settings["store_procent_of_passed_images"]
    if manual_quality_settings["starfinder_threshold"] is not None:
        blur_function = "starquality" #highest prio
        #sharper is greater
        persentile_side = 100.0 - manual_quality_settings["store_procent_of_passed_images"]
        blur_quality_list = deque([manual_quality_settings["starfinder_threshold"]], maxlen=1000)
    elif manual_quality_settings["blur_threshold"] is not None:
        blur_function = "blurmetric"
        #sharper goes positiv to zero
        persentile_side = manual_quality_settings["store_procent_of_passed_images"]
        blur_quality_list = deque([1.0], maxlen=1000)
    elif manual_quality_settings["laplace_threshold"] is not None:
        blur_function = "laplace"
        #sharper is greater
        persentile_side = 100.0 - manual_quality_settings["store_procent_of_passed_images"]
        blur_quality_list = deque([manual_quality_settings["laplace_threshold"]], maxlen=1000)

    print("Use metric: " + blur_function + " for image selection decision.")
    
        

    for gain in manual_quality_settings["use_gains"]:
        for exposure in manual_quality_settings["use_exposures"]:

            # fps req. for the given expsoure
            fps_exposure = 1e6 / exposure

            # do camera settings
            config["AnalogueGain"] = gain
            config["ExposureTime"] = int(exposure)
            config["FrameRate"] = min(fps_exposure, raw_camera_settings["max_fps"])

            print("Stard recording with Gain: {} and Exposure time: {} ms".format(config["AnalogueGain"], config["ExposureTime"] / 1e3))

            picamera_context.set_controls(config)
            time.sleep(0.4)

            try:
                picamera_context.start()
            except:
                continue
            st = datetime.timestamp(datetime.now())
            for i in range(manual_quality_settings["analyse_frames_per_settings"] + 1):

                if picamera_setsaveexit:
                    break

                buf, meta_data = picamera_context.capture_buffers(["raw", "main"])
                # first image has sometimes not the desired settings
                if i == 0:
                    continue

                while True:
                    try:
                        blur_quality_list.append(results_queue.get(block = False)[blur_function])
                        images_processed += 1
                    except queue.Empty:
                        break

                new_threshold = np.percentile(np.array(blur_quality_list), persentile_side)

                #print(video_queue.qsize())

                buffer_img.append( (buf[0], buf[1], new_threshold, blur_function) )
                if len(buffer_img) >= session["images_per_process"]:
                    video_queue.put( buffer_img )
                    buffer_img = []
            #   video_path_queue.put( (tmp_video_path, new_threshold, blur_function) )

            picamera_context.stop()

            if picamera_setsaveexit:
                break

        if picamera_setsaveexit:
            break
            
    
    for _ in range(len(procs)):
        video_queue.put( [(None, None, None, None)] )
        
    #wait for all ongoing threads
    for proc in procs:
        # workaround to avoid hanging queue because of the large size used here
        results_queue.cancel_join_thread()

        while proc.is_alive():
            st = datetime.timestamp(datetime.now())
            # get the last data from queue
            while True:
                try:
                    st = datetime.timestamp(datetime.now())
                    blur_quality_list.append(results_queue.get(block = False)[blur_function])
                    images_processed += 1
                    if PRINT_TIME_PROFILING:
                        print("queue get", datetime.timestamp(datetime.now()) - st)
                    #print("Add element")
                except queue.Empty:
                    break
            if PRINT_TIME_PROFILING:
                print("make list from queue", datetime.timestamp(datetime.now()) - st)

            try:
                proc.join(timeout = 0.1)
            except:
                print("Join fails", proc.is_alive())
                break

    st = datetime.timestamp(datetime.now())

    # resultslist will be empty here... 
    end_time = datetime.timestamp(datetime.now())

    print("{:.1f} images per second blur processed ({} / {} s)".format(images_processed / (end_time - start_time), 
                                                                images_processed, (end_time - start_time)))
