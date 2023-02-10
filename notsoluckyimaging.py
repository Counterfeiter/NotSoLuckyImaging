#!/usr/bin/python3
from engineio.payload import Payload
import time
import os
import signal
import sys
import glob
import re
import numpy as np
from datetime import datetime
from multiprocessing import Process, Pool, Queue
import queue  # not process save
import matplotlib.pyplot as plt
import __main__ as main
import atexit
from collections import deque  # not process save
import subprocess
import threading
from io import BytesIO
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO

from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

# image functions
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

WEBSERVER_PORT = 8081

session = {
    "object_name": "Star",
    "save_image_dng": False,
    "save_image_fits": True,
    # "img_folder": "/home/pi/luckyimaging/",
    # "img_folder" : "/media/pi/LOG_DONGLE/",
    "img_folder": "/media/pi/astro",
    "tmp_folder": "/tmp/",
    "num_cpus": 4,
    "images_per_process": 1
}

raw_camera_settings = {
    "size": (1920, 1080),
    "size_processing": (800, 600),
    "bayerformat": "SRGGB12",
    "bit_depth": 12,
    "max_fps": 5,
    "analoggain": (1, 31)  # (min, max)
}

manual_quality_settings = {
    "store_procent_of_passed_images": 3,
    "analyse_frames_per_settings": 20,
    "min_dynamic_range": 1,  # %
    # allow three pixels (dead pixels?) to be saturated
    "percentile": (0.1, 100.0 - 3.0 / (raw_camera_settings["size"][0] * raw_camera_settings["size"][1]) * 100.0),
    "use_eval_function": "laplace",
    # higher is better, no maximum
    "laplace_threshold": 20,
    # lower is better, range from 0 to 1.0
    "blur_threshold": 1.0,
    "starfinder": {
        "threshold": 8.0,
        "platesolve": False,

        "path_exec_solvefield": "/usr/bin/solve-field",
        # faster plate solving if given
        # "ra" : "05:41:03",
        "ra": "",
        # "dec" : "-01:55:06",
        "dec": "",
        "scale":
        {
            "enabled": True,
            "unit": "arcsecperpix",  # or "arcsecperpix" or "arcminwidth"
            "low": 17.0,
            "high": 18.1
        },
        # limit time to spend on plate solving the images
        "cputime": 10  # sec
    },
    "use_gain": 1,  # unity gain at about 11, use > 14 to have lower read noise
    "use_exposure": 10e3
}

config = {
    "FrameRate": raw_camera_settings["max_fps"],
    "ExposureTime": int(1e6 // raw_camera_settings["max_fps"]),
    "AnalogueGain": 1.0,
    # "ScalerCrop": offset + size # roi could be used later
}

# hack to disable satturation discards
max_adc_range = 5000  # 2**raw_camera_settings["bit_depth"] - 1
pixel_cnt = raw_camera_settings["size"][0] * raw_camera_settings["size"][1]

picamera_setsaveexit = None

enable_recording = False

# use to guide the telescope over indi in the main loop
guide_pulse = {"dir": "N",
               "time": 0
               }

processed_images = deque(maxlen=5)

# flask app
app = Flask(__name__, template_folder='./templates')
app.config['SECRET_KEY'] = 'secret!'

# fix error with max payload
Payload.max_decode_packets = 500  # cfg.service.ENGINEIO_MAX_DECODE_PACKETS
socketio = SocketIO(app, async_mode="threading")


def calc_polar_alignment_err(axis_drift_arc_sec, axis_pointed, timedelta_min):
    # (timedelta * cos(phi) * axis_drift * 3600 ) / 4

    # return is position error in arc minutes
    return 12.0/np.pi * axis_drift_arc_sec / (timedelta_min * np.cos(axis_pointed))


def solvefield_out_to_coord(out):
    ret = {
        "ra": None,
        "dec": None,
        "wcs": None
    }
    for line in out.decode("utf-8").split("\n"):
        # split at ( and  )
        line_sp = re.split(r"[\(\)]", line)
        # print(line_sp)
        if len(line_sp) > 4 and \
            "Field center:" in line_sp[0] and \
                "deg." in line_sp[4]:
            radec = line_sp[3].split(", ")
            try:
                ret["ra"] = float(radec[0])
                ret["dec"] = float(radec[1])
            except:
                pass

        # split at whitespaces
        line_sp = line.split(":")
        if len(line_sp) >= 2 and "Field rotation angle" in line_sp[0]:
            for word in line_sp[1].split(" "):
                try:
                    ret["wcs"] = float(word)
                except:
                    pass

    for key, value in ret.items():
        if value is None:
            return None

    return ret  # all coordinates could be parsed


def process_frames(process_id, video_queue, results):

    while True:
        # st = datetime.timestamp(datetime.now())
        for buf, gray, params in video_queue.get():

            # if PRINT_TIME_PROFILING:
            #    print("video_queue.get", datetime.timestamp(datetime.now()) - st)

            if buf is None:
                print("Close thread - work done")
                sys.exit(0)

            threshold, metric = params["threshold"], params["blurfunction"]

            try:
                arr = np.frombuffer(buf, dtype=np.uint16).reshape(
                    (raw_camera_settings["size"][1], raw_camera_settings["size"][0]))
            except Exception as e:
                print(e)
                continue

            try:
                arr_rgb = np.frombuffer(gray, dtype=np.uint8).reshape(
                    (raw_camera_settings["size_processing"][1], raw_camera_settings["size_processing"][0], 3))
                gray_image = rgb2gray(arr_rgb.astype(np.float32))
            except Exception as e:
                try:
                    # try if it is still a gray scale image or one layer
                    gray_image = np.frombuffer(gray, dtype=np.uint8).reshape(
                        (raw_camera_settings["size_processing"][1], raw_camera_settings["size_processing"][0]))
                except Exception as e:
                    print(e)
                    continue

            st = datetime.timestamp(datetime.now())
            min_value, max_value = np.percentile(
                arr, manual_quality_settings["percentile"])
            if PRINT_TIME_PROFILING:
                print("percentile", datetime.timestamp(datetime.now()) - st)

            range_percent = (max_value - min_value) / max_adc_range * 100

            image_passed = False
            laplaceout = None
            blur_metric = None
            star_quality = 0.0
            pos_dict = None

            saturation = False
            if max_value < max_adc_range:

                # plt.imshow(gray_image)
                # plt.show()

                image_passed = True

                # first blur test option - laplace kernel
                if metric == "laplace":
                    st = datetime.timestamp(datetime.now())
                    laplaceout = np.std(laplace(gray_image))
                    if laplaceout < manual_quality_settings["laplace_threshold"]:
                        image_passed = False

                # second blur test option - skikit image
                if metric == "blurmetric":
                    st = datetime.timestamp(datetime.now())
                    blur_metric = blur_effect(gray_image)
                    if blur_metric > manual_quality_settings["blur_threshold"]:
                        image_passed = False
                    if PRINT_TIME_PROFILING:
                        print("blur_effect", datetime.timestamp(
                            datetime.now()) - st)

                if metric == "starquality":
                    st = datetime.timestamp(datetime.now())
                    mean, median, std = sigma_clipped_stats(
                        gray_image, sigma=3.0)
                    daofind = DAOStarFinder(
                        fwhm=3.0, threshold=2.5*std, exclude_border=True)
                    try:
                        sources = daofind(gray_image - median)
                    except:
                        pass  # error if no stars are found
                    else:
                        if sources is not None:
                            sources = np.sort(sources, order="mag")[::-1]
                            mean = 0.0
                            # print(sources)
                            for src in sources[:10]:
                                mean += src["sharpness"]

                            if len(sources) > 0:
                                star_quality = mean / len(sources)

                            print("{} stars found in image!".format(len(sources)))

                            if manual_quality_settings["starfinder"]["platesolve"]:
                                hdul = fits.HDUList()
                                hdul.append(fits.PrimaryHDU())
                                hdul.append(fits.BinTableHDU(sources))
                                xyfitsfilepath = os.path.join(
                                    session["tmp_folder"], "solver_{}.xyls".format(str(process_id)))
                                hdul.writeto(xyfitsfilepath, overwrite=True)

                                sub_proc_call_str = ("solve-field \"{}\" -D \"{}\" --continue --overwrite --no-plots -w {:d} -e {:d} -X xcentroid -Y " +
                                                     "ycentroid -s mag --cpulimit {}").format(
                                    xyfitsfilepath, session["tmp_folder"],
                                    gray_image.shape[1], gray_image.shape[0],
                                    str(
                                        manual_quality_settings["starfinder"]["cputime"])
                                )
                                if manual_quality_settings["starfinder"]["ra"]:
                                    sub_proc_call_str += " --ra {}".format(
                                        manual_quality_settings["starfinder"]["ra"])
                                if manual_quality_settings["starfinder"]["dec"]:
                                    sub_proc_call_str += " --dec {}".format(
                                        manual_quality_settings["starfinder"]["dec"])
                                sub_proc_call_arr = sub_proc_call_str.split(
                                    " ")

                                if manual_quality_settings["starfinder"]["scale"]["enabled"]:
                                    sub_proc_call_str += " -L {}".format(
                                        manual_quality_settings["starfinder"]["scale"]["low"])
                                    sub_proc_call_str += " -H {}".format(
                                        manual_quality_settings["starfinder"]["scale"]["high"])
                                    sub_proc_call_str += " -u {}".format(
                                        manual_quality_settings["starfinder"]["scale"]["unit"])

                                # print(sub_proc_call_str)

                                proc = subprocess.Popen(
                                    sub_proc_call_str, shell=True, stdout=subprocess.PIPE)
                                out, err = proc.communicate()
                                # print(out)
                                pos_dict = solvefield_out_to_coord(out)

                                # print(pos_dict)

                    if PRINT_TIME_PROFILING:
                        print("DAOStarFinder", datetime.timestamp(
                            datetime.now()) - st)

                star_string = "{:.3f}".format(star_quality)
                blur_str = "{:.3f}".format(
                    blur_metric) if blur_metric is not None else "None"
                laplace_str = "{:.1f}".format(
                    laplaceout) if laplaceout is not None else "None"

                print("Min: {:.0f} Max: {:.0f}, Laplace: {} Blur: {} Starsharpness: {} {}".format(min_value, max_value,
                                                                                                  laplace_str, blur_str, star_string, "*" if image_passed else ""))
            else:
                print("Sensor saturation - skip")
                saturation = True

            out = {
                "config": config,
                "min": min_value,
                "max": max_value,
                "laplace": laplaceout,
                "blurmetric": blur_metric,
                "starquality": star_quality,
                "pass": image_passed,
                "saturation": saturation,
                "ts": datetime.now(),
                "gray_image": gray_image
            }

            if pos_dict:
                out["coords"] = pos_dict

            if results is None:
                print(out)
                return
            # send calulated results to main thread
            results.put(out)

            if image_passed and params["recording"]:
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
        res_dict[metric], metric[0], session["object_name"], int(
            res_dict["config"]["AnalogueGain"]),
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
    t.set(Tag.PhotometricInterpretation,
          PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, raw_camera_settings["bit_depth"])
    t.set(Tag.CFARepeatPatternDim, [2, 2])
    t.set(Tag.CFAPattern, CFAPattern.RGGB)
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
    t.set(Tag.Model, "IMX290C")
    r.options(t, path="", compress=False)
    r.convert(res_dict["raw"], filename=os.path.join(
        session["img_folder"], filename_dng))

    print(filename_dng + " saved!")


def writeFITSFile(res_dict, metric):
    filename_fits = "{:.3f}{}_{}_G{:02d}_E{:04d}_{:f}.fits".format(
        res_dict[metric], metric[0], session["object_name"], int(
            res_dict["config"]["AnalogueGain"]),
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
    # print(hdu.header)
    hdu.writeto(os.path.join(session["img_folder"], filename_fits))

    print(filename_fits + " saved!")

# Flask req. functions


def send_img_frame_to_webserver():
    while True:
        if len(processed_images) > 0:
            # try:
            byte_io = BytesIO()
            # print(processed_images[-1]["gray_image"].shape)
            pil_img = Image.fromarray(
                np.uint8(processed_images[-1]["gray_image"]), 'L')
            pil_img.save(byte_io, 'JPEG')
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + byte_io.getvalue() + b'\r\n')
            # except Exception as e:
            #     pass


@socketio.on('exposure_input')
def handle_sio_exposure_input(data):
    try:
        manual_quality_settings["use_exposure"] = float(data["value"])
    except:
        pass
    # print('received message: ' + str(data))


@socketio.on('startstop_button')
def handle_sio_startstop_button(data):
    global enable_recording
    try:
        enable_recording = True if "Start" in data["value"] else False
        print("Recording ", enable_recording)
    except:
        pass
    # print('received message: ' + str(data))


@socketio.on('guidepulse')
def handle_sio_guide_command(data):
    global guide_pulse
    try:
        guide_pulse["dir"] = data["dir"]
        guide_pulse["time"] = float(data["time"])
        print("Guide ", data)
    except:
        pass


@socketio.on('gain_input')
def handle_sio_gain_input(data):
    try:
        manual_quality_settings["use_gain"] = float(data["value"])
    except:
        pass
    # print('received message: ' + str(data))


@socketio.on('function_selection')
def handle_sio_function_selection(data):
    try:
        if data["value"] == "Laplace":
            manual_quality_settings["use_eval_function"] = "laplace"
        elif data["value"] == "StarQuality":
            manual_quality_settings["use_eval_function"] = "starquality"
        elif data["value"] == "BlurFunction":
            manual_quality_settings["use_eval_function"] = "blurmetric"
    except:
        pass
    # print('received message: ' + str(data))


@app.route('/live_feed')
def live_feed():
    return Response(send_img_frame_to_webserver(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/requests', methods=['POST', 'GET'])
def gui_task():
    return render_template('index.html')


cnt_exits = 0


def ctlC_handler(signum, frame):
    global picamera_setsaveexit
    global cnt_exits
    print("Exit by Crtl + C requested... ")
    picamera_setsaveexit = True
    cnt_exits += 1
    if cnt_exits > 2:
        exit(1)  # mybe the libcamera driver will fail in the next call


if __name__ == "__main__":
    picamera_setsaveexit = False
    signal.signal(signal.SIGINT, ctlC_handler)

    # create a thread save list for the results
    results_queue = Queue()
    video_queue = Queue(session["num_cpus"])
    # video_path_queue = Queue(3)
    blur_quality_list = deque([0.0], maxlen=1000)
    images_processed = 0

    threading.Thread(target=lambda: socketio.run(
        app, host='0.0.0.0', port=WEBSERVER_PORT, use_reloader=False)).start()

    try:
        # it is okay to not have pyindi_client installed - disable guiding
        from pyindi_guider import LX200MountGuide

        guider = LX200MountGuide("localhost", 7624)
    except Exception as e:
        print(e)
        guider = None

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        test_img = skimage.io.imread(test_file).astype(np.uint8)
        # test_img = np.swapaxes(test_img, 0, 1)
        if test_img.shape[2] >= 3:
            test_img = test_img[:, :, :3]

        # plt.imshow(test_img)
        # plt.show()

        params = {
            "threshold": 1.0,
            "blurfunction": "blurmetric",
            "recording": False
        }

        # print(test_img.shape)
        raw_camera_settings["size_processing"] = (
            test_img.shape[1], test_img.shape[0])  # trick gray scale image size
        test_raw = np.ones(
            (raw_camera_settings["size"][1], raw_camera_settings["size"][0]), dtype=np.uint16)
        raw_camera_settings["min_dynamic_range"] = 0.0  # trick dynamic
        video_queue.put([(test_raw.tobytes(), test_img.tobytes(), params)])
        process_frames(0, video_queue, None)

        exit(0)

    try:
        from cameras.picamera2 import PiCamera2
        camera = PiCamera2(
            raw_camera_settings["size"], raw_camera_settings["size_processing"])
    except:
        from cameras.folderloopcamera import FolderLoopCamera
        camera = FolderLoopCamera(
            raw_camera_settings["size"], list(raw_camera_settings["size_processing"]), scanfolder="/home/basti/Documents/Astro/OrionTrapez/lights")
        # from cameras.basecamera import BaseCamera
        # camera = BaseCamera(
        #     raw_camera_settings["size"], list(raw_camera_settings["size_processing"]) + [3])

    procs = []
    for i in range(session["num_cpus"]):
        procs.append(Process(target=process_frames,
                     args=(i, video_queue, results_queue)))
        procs[-1].start()

    start_time = datetime.timestamp(datetime.now())

    buffer_img = []

    persentile_side = manual_quality_settings["store_procent_of_passed_images"]
    blur_function = ""

    while picamera_setsaveexit == False:
        # fps req. for the given expsoure
        fps_exposure = 1e6 / manual_quality_settings["use_exposure"]

        # do camera settings
        config["AnalogueGain"] = int(manual_quality_settings["use_gain"])
        config["ExposureTime"] = int(manual_quality_settings["use_exposure"])
        config["FrameRate"] = min(fps_exposure, raw_camera_settings["max_fps"])

        print("Stard video stream with gain: {} and exposure time: {} ms".format(
            config["AnalogueGain"], config["ExposureTime"] / 1e3))

        camera.set_controls(config["AnalogueGain"],
                            config["ExposureTime"], config["FrameRate"])

        camera.start()

        while picamera_setsaveexit == False:
            socketio.sleep(0.01)  # yield

            if guider is not None and guide_pulse["time"] > 0:
                guider.guide(guide_pulse["dir"], guide_pulse["time"])
                guide_pulse["time"] = 0  # call only one per button click

            if blur_function != manual_quality_settings["use_eval_function"]:
                if manual_quality_settings["use_eval_function"] == "starquality":
                    # sharper is greater
                    persentile_side = 100.0 - \
                        manual_quality_settings["store_procent_of_passed_images"]
                    blur_quality_list = deque(
                        [manual_quality_settings["starfinder"]["threshold"]], maxlen=1000)
                elif manual_quality_settings["use_eval_function"] == "blurmetric":
                    # sharper goes positiv to zero
                    persentile_side = manual_quality_settings["store_procent_of_passed_images"]
                    blur_quality_list = deque([1.0], maxlen=1000)
                elif manual_quality_settings["use_eval_function"] == "laplace":
                    # sharper is greater
                    persentile_side = 100.0 - \
                        manual_quality_settings["store_procent_of_passed_images"]
                    blur_quality_list = deque(
                        [manual_quality_settings["laplace_threshold"]], maxlen=1000)

                blur_function = manual_quality_settings["use_eval_function"]

                print("Use metric: " + blur_function +
                      " for image selection decision.")

            if int(manual_quality_settings["use_gain"]) != config["AnalogueGain"] or \
                    int(manual_quality_settings["use_exposure"]) != config["ExposureTime"]:
                break

            st = datetime.timestamp(datetime.now())

            buf, meta_data = camera()

            while True:
                try:
                    processed_images.append(results_queue.get(block=False))
                    if not processed_images[-1]["saturation"]:
                        try:
                            blur_quality_list.append(
                                processed_images[-1][blur_function])
                        except:
                            pass
                        images_processed += 1
                except queue.Empty:
                    break

            new_threshold = np.percentile(
                np.array(blur_quality_list), persentile_side)

            params = {
                "threshold": new_threshold,
                "blurfunction": blur_function,
                "recording": enable_recording
            }

            buffer_img.append((buf[0], buf[1], params))
            if len(buffer_img) >= session["images_per_process"]:
                try:
                    video_queue.put(buffer_img, timeout=1)
                    buffer_img = []
                except:
                    pass

            if picamera_setsaveexit:
                break

        camera.stop()

    for _ in range(len(procs)):
        video_queue.put([(None, None, None)])

    # wait for all ongoing threads
    for proc in procs:
        # workaround to avoid hanging queue because of the large size used here
        results_queue.cancel_join_thread()

        while proc.is_alive():
            st = datetime.timestamp(datetime.now())
            # get the last data from queue
            while True:
                try:
                    st = datetime.timestamp(datetime.now())
                    results_queue.get(block=False)
                    images_processed += 1
                    if PRINT_TIME_PROFILING:
                        print("queue get", datetime.timestamp(
                            datetime.now()) - st)
                    # print("Add element")
                except queue.Empty:
                    break
            if PRINT_TIME_PROFILING:
                print("make list from queue",
                      datetime.timestamp(datetime.now()) - st)

            try:
                proc.join(timeout=0.1)
            except:
                print("Join fails", proc.is_alive())
                break

    st = datetime.timestamp(datetime.now())

    # resultslist will be empty here...
    end_time = datetime.timestamp(datetime.now())

    socketio.shutdown()

    print("{:.1f} images per second blur processed ({} / {} s)".format(images_processed / (end_time - start_time),
                                                                       images_processed, (end_time - start_time)))
