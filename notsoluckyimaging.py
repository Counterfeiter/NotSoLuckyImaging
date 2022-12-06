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

MATPLOTLIB_OUTPUT = False

PRINT_TIME_PROFILING = False

session = {
    "object_name": "Jupiter",
    "save_image_dng" : False,
    "save_image_fits" : True,
    #"img_folder": "/home/pi/luckyimaging/",
    "img_folder" : "/media/pi/LOG_DONGLE/",
    "tmp_folder": "/home/pi/ramdisk/",
    "num_cpus": 4
}

raw_camera_settings = {
    "size": (1920, 1080),
    "size_processing": (800, 600),
    "bayerformat": "SRGGB12",
    "bit_depth": 12,
    "max_fps": 60, #max 60 but short exposure videos get very fast large
    "analoggain": (1, 31) # (min, max)
}

manual_quality_settings = {
    "store_procent_of_passed_images": 1,
    "analyse_frames_per_settings": 100,
    "min_dynamic_range": 1, # %
    # allow three pixels (dead pixels?) to be saturated
    "percentile": (0.1, 100.0 - 3.0 / (raw_camera_settings["size"][0] * raw_camera_settings["size"][1]) * 100.0),
    #higher is better, no maximum
    "laplace_threshold": 1.0,
    # lower is better, range from 0 to 1.0
    "blur_threshold": None,
    "use_gains": [5],
    "use_exposures": [10e3] * 4
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


def process_raw_video(video_queue, results):

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

            if range_percent < manual_quality_settings["min_dynamic_range"]:
                print("Low dynamic range (min {:.1f}, max {:.1f}) detected - skip". format(min_value, max_value))
                continue

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

            # first blure test option - laplace kernel
            if manual_quality_settings["laplace_threshold"] is not None:
                st = datetime.timestamp(datetime.now())
                laplaceout = np.std(laplace(gray_image))
                if laplaceout < manual_quality_settings["laplace_threshold"]:
                    image_passed = False

                if PRINT_TIME_PROFILING:
                    print("np.std(laplace)", datetime.timestamp(datetime.now()) - st)

            # second blure test option - skikit image
            if manual_quality_settings["blur_threshold"] is not None:
                st = datetime.timestamp(datetime.now())
                blur_metric = blur_effect(gray_image)
                if blur_metric > manual_quality_settings["blur_threshold"]:
                    image_passed = False
                if PRINT_TIME_PROFILING:
                    print("blur_effect", datetime.timestamp(datetime.now()) - st)

            blur_str = "{:.3f}".format(blur_metric) if blur_metric is not None else "None"
            laplace_str = "{:.1f}".format(laplaceout) if laplaceout is not None else "None"

            print("Min: {:.0f} Max: {:.0f}, Laplace: {} Blur: {} {}".format(min_value, max_value , 
                        laplace_str, blur_str, "*" if image_passed else ""))

            out = {
                "config" : config,
                "min" : min_value,
                "max" : max_value,
                "laplace" : -laplaceout if laplaceout is not None else laplaceout,
                "blurmetric" : blur_metric,
                "pass" : image_passed,
                "ts" : datetime.now()
            }
            # send calulated results to main thread
            results.put(out)

            if image_passed:
                print(out[metric], " <=", threshold)
                if out[metric] <= threshold:
                    out["gray"] = gray_image
                    out["raw"] = arr
                    storeImages(out)

def storeImages(res):
    st = datetime.timestamp(datetime.now())

    if session["save_image_dng"]:
        writeDNGFile(res)
    if session["save_image_fits"]:
        writeFITSFile(res)

    if PRINT_TIME_PROFILING:
        print("store raw files", datetime.timestamp(datetime.now()) - st)

    return 0


def writeDNGFile(res_dict):
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
    blur_function = "blurmetric" if res_dict["blurmetric"] is not None else "laplace"
    filename_dng = "{:.3f}_{}_G{:02d}_E{:04d}_{:f}".format(
        res_dict[blur_function], session["object_name"], int(res_dict["config"]["AnalogueGain"]), 
        int(res_dict["config"]["ExposureTime"] // 1000), datetime.timestamp(res_dict["ts"]))
    r.convert(res_dict["raw"], filename=os.path.join(session["img_folder"], filename_dng ) )
    
    print(filename_dng + " saved!")

def writeFITSFile(res_dict):
    blur_function = "blurmetric" if res_dict["blurmetric"] is not None else "laplace"
    filename_fits = "{:.3f}_{}_G{:02d}_E{:04d}_{:f}.fits".format(
        res_dict[blur_function], session["object_name"], int(res_dict["config"]["AnalogueGain"]), 
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

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

if __name__ == "__main__":

    # create a thread save list for the results
    results_queue = Queue()
    video_queue = Queue(session["num_cpus"] * 5)
    # video_path_queue = Queue(3)
    blur_quality_list = deque([0.0], maxlen=1000)
    images_processed = 0

    # init camera device
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
                            main={"size": raw_camera_settings["size_processing"], "format": "RGB888"},
                            raw={"format": raw_camera_settings["bayerformat"], 'size': raw_camera_settings["size"]})
    picam2.configure(video_config)
    picam2.encode_stream_name = "raw"
    encoder = Encoder()
    

    procs = []
    for _ in range(session["num_cpus"]):
        procs.append(Process(target=process_raw_video, args=(video_queue, results_queue)))
        procs[-1].start()

    # sample_video_procs = []
    # for _ in range(session["num_cpus"]):
    #     sample_video_procs.append(Process(target=read_raw_video, args=(video_queue, video_path_queue)))
    #     sample_video_procs[-1].start()

    start_time = datetime.timestamp(datetime.now())

    results = []

    buffer_img = []

    blur_function = "blurmetric" if manual_quality_settings["blur_threshold"] is not None else "laplace"

    for gain in manual_quality_settings["use_gains"]:
        for exposure in manual_quality_settings["use_exposures"]:

            # fps req. for the given expsoure
            fps_exposure = 1e6 / exposure

            # do camera settings
            config["AnalogueGain"] = gain
            config["ExposureTime"] = int(exposure)
            config["FrameRate"] = min(fps_exposure, raw_camera_settings["max_fps"])

            print("Stard recording with Gain: {} and Exposure time: {} ms".format(config["AnalogueGain"], config["ExposureTime"] / 1e3))

            # record into a ramfs or tmpfs storage for speed
            tmp_video_path = os.path.join(session["tmp_folder"], "tmp" + str(datetime.timestamp(datetime.now())) + ".raw")
            tmp_pts_path = os.path.join(session["tmp_folder"], "timestamp.txt")

            picam2.set_controls(config)
            time.sleep(0.4)

            try:
                picam2.start()
            except:
                continue
            st = datetime.timestamp(datetime.now())
            for i in range(manual_quality_settings["analyse_frames_per_settings"] + 1):
                buf, meta_data = picam2.capture_buffers(["raw", "main"])
                if i == 0:
                    continue
                #arr = np.frombuffer(buf[0], dtype=np.uint16).reshape((raw_camera_settings["size"][1], raw_camera_settings["size"][0]))
                #arr_rgb = np.frombuffer(buf[1], dtype=np.uint8).reshape((raw_camera_settings["size_processing"][1], raw_camera_settings["size_processing"][0], 3))
                #arr_gray = rgb2gray(arr_rgb)

            # print("FPS", 100.0 / (datetime.timestamp(datetime.now()) - st) )
            # print(arr.shape, len(buf[1]))
            # print(meta_data)

            # plt.imshow(arr_gray, cmap='gray')
            # plt.show()

            # exit()
            # we need this delay to give the libcamera driver some time for the next record
            # else we see an IO error
            # time.sleep(0.5)
            # picam2.start_recording(encoder, tmp_video_path, pts=tmp_pts_path)
            # # for small exposures record min two seconds to grab enouth frames
            # time.sleep(max(2, (manual_quality_settings["analyse_frames_per_settings"] + 2) * 1/fps_exposure))
            # picam2.stop_recording()

                while True:
                    try:
                        blur_quality_list.append(results_queue.get(block = False)[blur_function])
                        images_processed += 1
                    except queue.Empty:
                        break

                new_threshold = np.percentile(np.array(blur_quality_list), manual_quality_settings["store_procent_of_passed_images"])

                #print(video_queue.qsize())

                buffer_img.append( (buf[0], buf[1], new_threshold, blur_function) )
                if len(buffer_img) > 3:
                    video_queue.put( buffer_img )
                    buffer_img = []
            #   video_path_queue.put( (tmp_video_path, new_threshold, blur_function) )

            picam2.stop()
            
    
    for _ in range(len(procs)):
        video_queue.put( [(None, None, None, None)] )

    # for proc in sample_video_procs:
    #     proc.join()
        
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



    if MATPLOTLIB_OUTPUT:
        # take the worst and the best to display the gray scale images

        bestseeing = min(results, key=lambda x:x[blur_function])
        worstseeing = max(results, key=lambda x:x[blur_function])

        hist_best, _ = np.histogram(bestseeing["gray"], 64, (0, max_adc_range))
        hist_worst, _ = np.histogram(worstseeing["gray"], 64, (0, max_adc_range))

        ### display grayscale images
        f, axarr = plt.subplots(2,2)
        best_im = bestseeing["gray"]
        best_im = skimage.exposure.rescale_intensity(best_im, out_range=(0, 255))
        print("Best blure", bestseeing['blurmetric'])
        axarr[0,0].imshow(best_im, cmap='gray', vmin=0, vmax=255)
        worst_img = worstseeing["gray"]
        worst_img = skimage.exposure.rescale_intensity(worst_img, out_range=(0, 255))
        print("Worst blure", worstseeing['blurmetric'])
        axarr[1,0].imshow(worst_img, cmap='gray', vmin=0, vmax=255)

        ### display historgrams
        x_histo = np.linspace(0, max_adc_range, len(hist_best), endpoint = False)
        axarr[0,1].plot(x_histo, hist_best)
        #axarr[0,1].set_yscale("log")
        axarr[1,1].plot(x_histo, hist_worst)
        #axarr[1,1].set_yscale("log")

    st = datetime.timestamp(datetime.now())

    # sorting is ineffetiv, get blur values to new list
    # blur_value_list = []
    # for res in results:
    #     blur_value_list.append(res[blur_function])

    # # sort list inplace
    # blur_value_list.sort(reverse=True if blur_function == "laplace" else False)

    # blur_value_list = blur_value_list[:int(round(float(len(blur_value_list)) * manual_quality_settings["store_procent_of_passed_images"] / 100))]

    # pool_list = []
    # for res in results:
    #     if res[blur_function] in blur_value_list:
    #         pool_list.append(res)
    #         #results_queue.put(res)

    # if PRINT_TIME_PROFILING:
    #     print("sorting for pool job", datetime.timestamp(datetime.now()) - st)

    # if 0:
    #     for res in results:

    #         # debug output - save all gray scale images as png and name it with the blure metric
    #         # stretch the grayscale image to display it later

    #         gray_image = skimage.exposure.rescale_intensity(res["gray"], out_range=(100, max_adc_range-100))
    #         try:
    #             png_path = '/home/pi/ramdisk/pngs/{:.2f}.png'.format(res[blur_function])
    #             plt.imsave(png_path, gray_image, cmap='gray', vmin=0, vmax=max_adc_range)
    #         except Exception as e:
    #             print(e)

    # len_results = len(results)
        
    # with Pool(session["num_cpus"]) as p:
    #     p.map(storeImages, pool_list)

    # resultslist will be empty here... 
    end_time = datetime.timestamp(datetime.now())

    print("{:.1f} images per second blur processed ({} / {} s)".format(images_processed / (end_time - start_time), 
                                                                images_processed, (end_time - start_time)))

    if MATPLOTLIB_OUTPUT:
        plt.show()
