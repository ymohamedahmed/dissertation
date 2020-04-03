from biosppy import storage
import biosppy
import pandas as pd
import heartpy as hp
from os import listdir
from os.path import isfile, join, isdir
from region_selection import KMeans, IntervalSkinDetector, PrimitiveROI, BayesianSkinDetector
from face_det import KLTBoxingWithThresholding, DNNDetector, RepeatedDetector
from hr_isolator import ICAProcessor, PCAProcessor, Processor, PrimitiveProcessor
from biosppy.signals import ecg as ECG
import pyedflib
import numpy as np
from pipeline import tracking_pipeline
from configuration import Configuration, PATH, MAC_PATH, LINUX_PATH, MAC
from numpy import genfromtxt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter
import cv2 as cv
import time
import csv

DEBUG = False
def check_path(file):
    if MAC:
        return file.replace(LINUX_PATH, MAC_PATH)
    else: 
        return file.replace(MAC_PATH, LINUX_PATH)

def get_ecg_signal(file_path):
    file_path = check_path(file_path)
    f = pyedflib.EdfReader(file_path)
    signal_labels = f.getSignalLabels()
    # ECG is labelled different for mahnob and for the ECG device
    label = "ECG" if file_path.endswith("edf") else "EXG1"
    ecg = signal_labels.index(label)
    sampling_freq = f.getSampleFrequency(ecg)
    ecg = f.readSignal(ecg)
    ecg = ecg[start_ecg_point(ecg):]
    ecg = (ecg-np.mean(ecg))/np.std(ecg)
    f._close()
    return (ecg, sampling_freq)

def get_avi_bdf(folder_path):
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))] 
    avi = [f for f in onlyfiles if f.endswith(".avi")]
    bdf = [f for f in onlyfiles if f.endswith(".bdf")]
    return avi[0], bdf[0]

def start_ecg_point(signal):
    mean = np.mean(signal)
    cleaned = (signal-mean)<0.5*np.std(signal)
    return np.argmax(cleaned)

def get_ppg_signal(file_path):
    return pd.read_csv(check_path(file_path))

def upsample(data, low, high):
    x = np.arange(low, high, 1/1000)
    y = np.interp(x, data["Time"], data["PPG"])

    y = hp.filter_signal(y, [0.7, 3.5], sample_rate=1000, 
                            order=3, filtertype='bandpass')
    return y

def add_time_to_ppg(data):
    t_max, t_min = np.max(data["Timestamp"]),np.min(data["Timestamp"])
    data["Time"] = 60*(data["Timestamp"]-t_min)/(t_max-t_min)
    return data

def mean_heart_rate(signal, sampling_freq):
    try:
        signal = hp.filter_signal(signal, [0.7, 3.5], sample_rate=sampling_freq, 
                        order=3, filtertype='bandpass') 
        _, m = hp.process(hp.scale_data(signal), sampling_freq)
        return m["bpm"]
    except Exception as e:
        print(e)
        return None

def track_ppg(video_path, config:Configuration):
    cap = cv.VideoCapture(video_path)
    values = np.array([])
    times = np.array([])
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_number = 0
    framerate = int(cap.get(cv.CAP_PROP_FPS))

    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            cap.release()
            cv.destroyAllWindows()
            break
        face_found = False
        faces, cropped = None, None
        while(not(face_found)):
            faces, frame, cropped, profiling = config.tracker.track(frame)
            if(len(faces) > 0):
                face_found = True
            else:
                ret, frame = cap.read()
                values = np.append(values, [np.nan, np.nan, np.nan])
                if ret == False:
                    cap.release()
                    cv.destroyAllWindows()
                    break

        frame_number += 1
        x,y,w,h = faces[0]
        
        start = time.time()
        area_of_interest, value = config.region_selector.detect(cropped)
        end = time.time()
        values = np.append(values, value)
        times = np.append(times, end-start)
    values = values.reshape(len(values)//3, 3)
    length,_ = values.shape
    # Interpolate none values
    xp = np.arange(length)
    for i in range(3):
        nan_indices = xp[np.isnan(values[:,i])]
        nans = np.isnan(values[:,i])
        nan_indices = xp[nans]
        values[nan_indices,i] = np.interp(nan_indices, xp[~nans], values[~nans,i]) 
    return values, framerate, np.mean(times), np.std(times)

def test_data():
    # in order video file, ppg, ecg
    files =  []
    base_path = "experiments/yousuf-re-run/"
    for dist in ["1", "1.5", "2"]:
        for exer in ["stat", "star", "jog"]:
            for repeat in range(1,4):
                file = f"{base_path}{dist}_{exer}_{repeat}"
                files.append([f"{PATH}{file}.mp4", f"{PATH}{file}.csv", f"{PATH}{file}.edf"])
    mahnob_path = f"{PATH}mahnob/Sessions/"
    for fold in listdir(mahnob_path): 
        vid, bdf = get_avi_bdf(mahnob_path + fold)
        files.append([f"{mahnob_path}{fold}/{vid}", None, f"{mahnob_path}{fold}/{bdf}"])
    return files
                
def hr_with_max_power(freqs):
    # return max(freqs, key=itemgetter(1))[0]
    hrs = [freqs[0], freqs[3], freqs[6]]
    return hrs[np.argmax([freqs[1], freqs[4], freqs[7]])]

def noise(data, framerate):
    data = (data-np.mean(data))/np.std(data)
    transform = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1.0/framerate)
    freqs = 60*freqs
    band_pass = np.where((freqs < 40) | (freqs > 240) )[0]
    transform = np.abs(transform)**2
    noise = np.sum(transform[band_pass])
    return noise

def map_config(config: list, window_size, offset):
    """
        Take a list of three numbers and return a configuration
        Configurations based on the following:
        tracker in {RepeatedDetector, KLTBoxingWithThresholding}
        region_selector in {PrimitiveROI, IntervalThresholding, BayesianSkinDetector(weighted=False), BayesianSkinDetector(weighted=True)}
        signal_processor in {PCA, ICA}
    """
    trackers = [RepeatedDetector(DNNDetector()), KLTBoxingWithThresholding(DNNDetector(), recompute_threshold=0.15), KLTBoxingWithThresholding(DNNDetector())]
    region_selectors = [PrimitiveROI(), IntervalSkinDetector(), BayesianSkinDetector(weighted=False), BayesianSkinDetector(weighted=True)]
    signal_processor = [PCAProcessor(), ICAProcessor()]
    t, rs, sp = config[0], config[1], config[2]
    return Configuration(trackers[t], region_selectors[rs], signal_processor[sp], window_size, offset)

def write_ppg_out(files, ppg_meta_output):
    meta_columns = ["Video file", "rPPG file", "PPG file", "ECG file", "Framerate", "Tracker", "Region selector", "Time mean", "Time std", "Number of frames", "Total time", "Noise"]
    with open(ppg_meta_output, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(meta_columns)

    for index, file_set in enumerate(files):
        for tr in range(3):
            for rs in range(4):
                vid, ppg_file, ecg_file = file_set[0], file_set[1], file_set[2]
                if vid.endswith(".avi"):
                    config = map_config([tr, rs, 0], 0, 0)
                    print("========================")
                    print(f"Experiments completion: {100*((12*index) + (4*tr) + rs)/(12*len(files))}%")
                    print(f"Video: {vid}, PPG file: {ppg_file}, ECG file: {ecg_file}, Tracker: {tr}, Region selector: {rs}")
                    start = time.time()
                    values, mean_time, time_std, framerate = track_ppg(vid, config)
                    total = time.time()-start
                    value_output = f"{PATH}output/hr_evaluation/{vid.split('/')[-1][:-4]}-{str(config.tracker)}-{str(config.region_selector)}.csv"
                    meta_row = [vid, value_output, ppg_file, ecg_file, framerate, str(config.tracker), str(config.region_selector), mean_time, time_std, len(values), total, noise(values, framerate)]
                    with open(ppg_meta_output, 'a') as fd:
                        writer = csv.writer(fd)
                        writer.writerow(meta_row)
                    np.savetxt(value_output, values)

def enough_ppg_samples(ppg_signal):
    n_rows = len(ppg_signal["Time"])
    if ppg_signal.empty or n_rows < 1000:
        return False

    n_bins = 10
    hist, bins = np.histogram(ppg_signal["Time"], bins=n_bins)
    # print(hist)
    threshold = 0.7*n_rows/n_bins
    if DEBUG: 
        print(f"Rows: {n_rows}")
        print(threshold)
        print(hist)
        print(hist > threshold)
        plt.plot(ppg_signal["Time"], ppg_signal["PPG"])
        plt.show()
    # lower, upper = np.mean(hist)-np.std(hist), np.mean(hist)+np.std(hist)
    # return all((hist>lower)&(hist<upper))
    return all(hist > threshold)

def evaluate(rppg_signal, ppg_file, ecg_file, window_size, offset, framerate):
    rppg_hr_pca = np.array([])
    rppg_hr_ica = np.array([])
    rppg_hr_rgb = np.array([])

    ecg, ecg_sf = get_ecg_signal(ecg_file)
    ecg_ws, ecg_o = len(ecg)*window_size/len(rppg_signal), len(ecg)*offset/len(rppg_signal)
    ecg_hr = []

    ppg_sf = 1000
    print(ppg_file)
    ppg_file_exists = not(ppg_file is None) and not(ppg_file == "") if type(ppg_file) == str else not(np.isnan(ppg_file)) #and not(np.isnan(ppg_file))
    if ppg_file_exists:
        ppg = add_time_to_ppg(get_ppg_signal(ppg_file))
    ppg_ws, ppg_o = 60.0*window_size/len(rppg_signal), 60.0 * offset/len(rppg_signal)
    ppg_hr = []

    for i, base in enumerate(np.arange(0, len(rppg_signal)-window_size+1, offset)):
        sig = rppg_signal[base:base+window_size]
        rppg_hr_pca = np.append(rppg_hr_pca, PCAProcessor().get_hr(sig, framerate))
        rppg_hr_ica = np.append(rppg_hr_ica, ICAProcessor().get_hr(sig, framerate))
        rppg_hr_rgb = np.append(rppg_hr_rgb, PrimitiveProcessor().get_hr(sig, framerate))

        e_low, e_high = int(i*ecg_o), int((i*ecg_o)+ecg_ws)
        try:
            ecg_hr.append(mean_heart_rate(ecg[e_low:e_high],ecg_sf))
        except Exception as e:
            print(e)
            ecg_hr.append(None)
            if DEBUG: 
                plt.plot(ecg[e_low:e_high])
                plt.show()

        p_low, p_high = int(i*ppg_o), int((i*ppg_o)+ppg_ws)
        
        if ppg_file_exists:
            filtered = ppg[(ppg["Time"] < p_high)&(ppg["Time"]>p_low)]
            # if (len(filtered) > window_size):
            if(enough_ppg_samples(filtered)):
                signal = upsample(filtered, p_low, p_high)
                if DEBUG: 
                    plt.plot(signal)
                    plt.show()
                try:
                    _, m = hp.process(hp.scale_data(signal), ppg_sf)
                    ppg_hr.append(m["bpm"])
                except Exception as e:
                    print("Error for PPG signal")
                    print(e)
                    ppg_hr.append(None)
                    if DEBUG:
                        plt.plot(ppg["Time"], ppg["PPG"])
                        plt.show()
                        plt.plot(signal)
                        plt.show()
            else: 
                ppg_hr.append(None)
        else: 
            ppg_hr.append(None)
    rppg_hr_ica = rppg_hr_ica.reshape((len(rppg_hr_ica)//9, 9))
    rppg_hr_pca = rppg_hr_pca.reshape((len(rppg_hr_pca)//9, 9))
    rppg_hr_rgb = rppg_hr_rgb.reshape((len(rppg_hr_rgb)//9, 9))
    return (rppg_hr_ica, rppg_hr_pca, rppg_hr_rgb, ppg_hr, ecg_hr)


def signal_processing_experiments(files, ppg_meta_file):
    sp_output = f"{PATH}output/hr_evaluation/sp-with-beat-counting-threshold-0.09.csv"
    ppg_meta_file = check_path(ppg_meta_file)
    columns = ["Video", "Tracker", "Region selector", "Window size", "Offset size", "Heart Rate Number", 
    "rPPG HR ICA", "rPPG HR PCA", "PPG HR", "ECG HR", 
     "ICA 1 HR", "ICA 1 Power", "ICA 1 BC",
     "ICA 2 HR", "ICA 2 Power", "ICA 2 BC", "ICA 3 HR", "ICA 3 Power", "ICA 3 BC",
     "PCA 1 HR", "PCA 1 Power", "PCA 1 BC",
     "PCA 2 HR", "PCA 2 Power", "PCA 2 BC",
     "PCA 3 HR", "PCA 3 Power", "PCA 3 BC",
     "R HR", "R Power", "R BC",
     "G HR", "G Power", "G BC",
     "B HR", "B Power", "B BC"
     ]
    ppg_meta = pd.read_csv(ppg_meta_file)
    with open(sp_output, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(columns)
    #Correct for swapped time mean and framerate
    temp = ppg_meta["Framerate"]
    ppg_meta["Framerate"] = ppg_meta["Time mean"]
    ppg_meta["Time mean"] = temp
    for index, ppg_row in ppg_meta.iterrows():
        # if (ppg_row["Video file"].endswith(".avi")):
        signal = np.loadtxt(check_path(ppg_row["rPPG file"]))
        # for ws in np.arange(600, 1200, 100):
        #     for off in np.arange(30, 120, 30):
        ws, off = 600, 60
        print("===================================")
        progress = 100*index/len(ppg_meta)
        print(f"Experiment progress: {progress}%")
        vid_name = ppg_row["Video file"]
        print(f"Considering: {vid_name}, Window size: {ws}, Offset: {off}")
        rppg_ica, rppg_pca, rppg_rgb, ppg_hr, ecg_hr = evaluate(signal, ppg_row["PPG file"], ppg_row["ECG file"], ws, off, ppg_row["Framerate"])
        n_rows, _ = rppg_ica.shape
        for i in range(n_rows):
            # print(len(ppg_hr))
            # print(len(ecg_hr))
            # print(rppg_ica.shape)
            # print(rppg_pca.shape)
            # print(rppg_rgb.shape)
            row = [
                ppg_row["Video file"], ppg_row["Tracker"], ppg_row["Region selector"], ws, off, i, 
                hr_with_max_power(rppg_ica[i, :]), rppg_pca[i,0], ppg_hr[i], ecg_hr[i], 
                rppg_ica[i,0], rppg_ica[i,1], rppg_ica[i,2], 
                rppg_ica[i,3], rppg_ica[i,4], rppg_ica[i,5], 
                rppg_ica[i,6], rppg_ica[i,7], rppg_ica[i,8], 
                rppg_pca[i,0], rppg_pca[i,1], rppg_pca[i,2], 
                rppg_pca[i,3], rppg_pca[i,4], rppg_pca[i,5],
                rppg_pca[i,6], rppg_pca[i,7], rppg_pca[i,8],
                rppg_rgb[i,0], rppg_rgb[i,1], rppg_rgb[i,2], 
                rppg_rgb[i,3], rppg_rgb[i,4], rppg_rgb[i,5],
                rppg_rgb[i,6], rppg_rgb[i,7], rppg_rgb[i,8]
                ]
            with open(sp_output, 'a') as fd:
                writer = csv.writer(fd)
                writer.writerow(row)
            

if __name__ == "__main__":
    files = test_data()
    ppg_meta_output = f"{PATH}output/hr_evaluation/ppg_meta.csv"
    # write_ppg_out(files, ppg_meta_output)
    signal_processing_experiments(files, ppg_meta_output)
    pass


