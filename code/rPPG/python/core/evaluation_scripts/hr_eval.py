from biosppy import storage
import biosppy
import pandas as pd
import heartpy as hp
from os import listdir
from os.path import isfile, join, isdir
import sys
sys.path += ["/Users/yousuf/Workspace/dissertation/code/rPPG/python/core/"]
from region_selection import KMeans, IntervalSkinDetector, PrimitiveROI, BayesianSkinDetector
from face_det import KLTBoxingWithThresholding, DNNDetector, RepeatedDetector
from hr_isolator import ICAProcessor, PCAProcessor, Processor, PrimitiveProcessor
from biosppy.signals import ecg as ECG
import math
import pyedflib
import numpy as np
from pipeline import tracking_pipeline
from configuration import Configuration, PATH
from numpy import genfromtxt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter
import cv2 as cv
import time
import csv

DEBUG = False

def get_ecg_signal(file_path):
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
    return pd.read_csv(file_path)

def upsample(data, low, high):
    # x = np.arange(low, high, 1/1000)
    # y = np.interp(x, data["Time"], data["PPG"])
    sf = len(data)/(high-low)
    y = hp.filter_signal(data["PPG"], [0.7, 3.5], sample_rate=sf, 
                            order=3, filtertype='bandpass')
    return y

def add_time_to_ppg(data):
    t_max, t_min = np.max(data["Timestamp"]),np.min(data["Timestamp"])
    data["Time"] = 60*(data["Timestamp"]-t_min)/(t_max-t_min)
    return data

def mean_heart_rate(signal, sampling_freq):
    rpeaks = biosppy.ecg.engzee_segmenter(signal, sampling_rate=sampling_freq)[0]
    time = len(signal)/sampling_freq
    hr = 60*len(rpeaks)/time
    return hr

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
    return files
                
def hr_with_max_power(freqs):
    # return max(freqs, key=itemgetter(1))[0]
    hrs = [freqs[0], freqs[3], freqs[6]]
    return hrs[np.argmax([freqs[1], freqs[4], freqs[7]])]

def noise(data, framerate, true_hr):
    if true_hr is None:
        return None

    data = (data-np.mean(data))/np.std(data)
    transform = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1.0/framerate)
    freqs = 60*freqs
    delta = 4
    band_pass = np.where((freqs < 40) | (freqs > 240) )[0]
    transform[band_pass] = 0
    lower = (freqs > (true_hr + delta))
    upper = (freqs < (true_hr - delta))
    hr_range = np.where(lower | upper)[0]
    transform = np.abs(transform)**2
    noise_value = np.sum(transform[hr_range])
    numerator = np.sum(transform)-np.sum(transform[hr_range])
    return numerator/np.sum(transform)

def map_config(config: list, window_size, offset):
    """
        Take a list of three numbers and return a configuration
        Configurations based on the following:
        tracker in {RepeatedDetector, KLTBoxingWithThresholding}
        region_selector in {PrimitiveROI, IntervalThresholding, BayesianSkinDetector(weighted=False), BayesianSkinDetector(weighted=True)}
        signal_processor in {PCA, ICA}
    """
    def map_tracker(i):
        if i == 0: return RepeatedDetector(DNNDetector())
        elif i == 1: return KLTBoxingWithThresholding(DNNDetector(), recompute_threshold=0.15)
    
    def map_region_selector(i):
        if i == 0: return PrimitiveROI()
        elif i == 1: return IntervalSkinDetector()
        elif i == 2: return BayesianSkinDetector(weighted=False)
        elif i == 3: return BayesianSkinDetector(weighted=True)
    
    def map_signal_processor(i):
        if i == 1: return PCAProcessor()
        elif i == 2: return ICAProcessor()
    t, rs, sp = config[0], config[1], config[2]
    return Configuration(map_tracker(t), map_region_selector(rs), map_signal_processor(sp), window_size, offset)

def write_ppg_out(files, ppg_meta_output):
    meta_columns = ["Video file", "rPPG file", "PPG file", "ECG file", "Framerate", "Tracker", "Region selector", "Time mean", "Time std", "Number of frames", "Total time"]
    with open(ppg_meta_output, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(meta_columns)

    for index, file_set in enumerate(files):
        for tr in range(1,2):
            for rs in range(3,4):
                vid, ppg_file, ecg_file = file_set[0], file_set[1], file_set[2]
                config = map_config([tr, rs, 0], 0, 0)
                print("========================")
                print(f"Experiments completion: {100*((12*index) + (4*tr) + rs)/(12*len(files))}%")
                print(f"Video: {vid}, PPG file: {ppg_file}, ECG file: {ecg_file}, Tracker: {tr}, Region selector: {rs}")
                start = time.time()
                values, framerate, mean_time, time_std = track_ppg(vid, config)
                total = time.time()-start
                value_output = f"{PATH}output/hr_evaluation/{vid.split('/')[-1][:-4]}-{str(config.tracker)}-{str(config.region_selector)}-fixed.csv"
                meta_row = [vid, value_output, ppg_file, ecg_file, framerate, str(config.tracker), str(config.region_selector), mean_time, time_std, len(values), total]
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
    threshold = 0.7*n_rows/n_bins
    if DEBUG: 
        print(f"Rows: {n_rows}")
        print(threshold)
        print(hist)
        print(hist > threshold)
        plt.plot(ppg_signal["Time"], ppg_signal["PPG"])
        plt.show()
    return all(hist > threshold)

def sample_framerate(ppg, fr, true_fr):
    xs = np.arange(0, len(ppg), true_fr/fr)
    ups = np.zeros(shape=(len(xs), 3))
    for i in range(3):
        ups[:,i] = np.interp(xs, np.arange(0, len(ppg)), ppg[:,i])
    return ups


def evaluate(rppg_signal, ppg_file, ecg_file, window_size, offset, framerate, true_framerate):
    rppg_hr_ica = np.array([])

    ecg, ecg_sf = get_ecg_signal(ecg_file)
    noises = np.array([])
    selfd_noises = np.array([])
    ecg_ws, ecg_o = len(ecg)*window_size/len(rppg_signal), len(ecg)*offset/len(rppg_signal)
    ecg_hr = []
    ecg_hr_fft = []

    ppg_file_exists = not(ppg_file is None) and not(ppg_file == "") if type(ppg_file) == str else not(np.isnan(ppg_file)) #and not(np.isnan(ppg_file))
    if ppg_file_exists:
        ppg = add_time_to_ppg(get_ppg_signal(ppg_file))
        ppg_sf = len(ppg)/60.0
    ppg_ws, ppg_o = 60.0*window_size/len(rppg_signal), 60.0 * offset/len(rppg_signal)
    ppg_hr = []
    ppg_hr_fft = []

    for i, base in enumerate(np.arange(0, len(rppg_signal)-window_size+1, offset)):
        sig = rppg_signal[base:base+window_size]
        rppg_hr_ica = np.append(rppg_hr_ica, ICAProcessor().get_hr(sig, framerate))

        e_low, e_high = int(i*ecg_o), int((i*ecg_o)+ecg_ws)
        # ecg_hr_fft.append(Processor()._prevalent_freq(ecg[e_low:e_high], ecg_sf)[0])
        ecg_hr_fft.append(None)
        try:
            ecg_hr.append(mean_heart_rate(ecg[e_low:e_high],ecg_sf))
        except Exception as e:
            print(e)
            ecg_hr.append(None)
            if DEBUG: 
                plt.plot(ecg[e_low:e_high])
                plt.show()

        noises = np.append(noises, np.array([noise(sig[:,dim], framerate, ecg_hr[-1]) for dim in range(3)]))
        selfd_noises = np.append(selfd_noises, np.array([noise(sig[:,dim], framerate, hr_with_max_power(rppg_hr_ica[i*9:])) for dim in range(3)]))
        
        if False:
            filtered = ppg[(ppg["Time"] < p_high)&(ppg["Time"]>p_low)]
            if(len(filtered) > window_size):
                hr_fft = Processor()._prevalent_freq(filtered["PPG"], ppg_sf)[0]
                ppg_hr_fft.append(hr_fft)
            else: 
                ppg_hr_fft.append(None)
            if(enough_ppg_samples(filtered)):
                signal = upsample(filtered, p_low, p_high)
                if DEBUG: 
                    plt.plot(signal)
                    plt.show()
                try:
                    _, m = hp.process(signal, ppg_sf)
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
            ppg_hr_fft.append(None)

    rppg_hr_ica = rppg_hr_ica.reshape((len(rppg_hr_ica)//9, 9))
    noises = noises.reshape(len(noises)//3, 3)
    selfd_noises = selfd_noises.reshape(len(selfd_noises)//3, 3)
    return (rppg_hr_ica, ppg_hr, ppg_hr_fft, ecg_hr, ecg_hr_fft, noises, selfd_noises)

def majority_vote(freqs):
    hrs = [freqs[0], freqs[3], freqs[6]]
    powers = [freqs[1], freqs[4], freqs[7]]
    i = np.argsort(powers)
    hr_max_power = hrs[i[-1]]
    mean_of_non_max = 1/2 * (np.sum(hrs)- hr_max_power)
    if abs(hr_max_power - mean_of_non_max)/hr_max_power > 0.3 and hr_max_power == max(hrs):
        return hrs[i[-2]]
    else:
        return hr_max_power

def signal_processing_experiments(files, ppg_meta_file, sp_output):
    columns = ["Video", "Tracker", "Region selector", "Window size", "Offset size", "Heart Rate Number", "Framerate", 
     "rPPG HR ICA", "rPPG HR MV", 
     "PPG HR BC", "PPG HR FFT",
     "ECG HR BC", "ECG HR FFT",
     "ICA 1 HR", "ICA 1 Power", "ICA 1 BC", "Noise 1", "SelfD Noise 1",
     "ICA 2 HR", "ICA 2 Power", "ICA 2 BC", "Noise 2", "SelfD Noise 2",
     "ICA 3 HR", "ICA 3 Power", "ICA 3 BC", "Noise 3", "SelfD Noise 3"
     ]
    ppg_meta = pd.read_csv(ppg_meta_file)
    with open(sp_output, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(columns)

    for index, ppg_row in ppg_meta.iterrows():
        rppg_file = ppg_row["rPPG file"]
        signal = np.loadtxt(rppg_file)
        ws, off = 600, 60
        print("===================================")
        progress = 100*index/len(ppg_meta)
        print(f"Experiment progress: {progress}%")
        vid_name = ppg_row["Video file"]
        print(f"Considering: {vid_name}, Window size: {ws}, Offset: {off}")
        if("KLTBoxingWithThresholding" in rppg_file and "BayesianSkinDetector-weighted" in rppg_file):
            for framerate in [5, 10, 15, 20, 25, 30]:
                print(f"Considering framerate: {framerate}")
                true_fr = ppg_row["Framerate"]
                new_signal = sample_framerate(signal, framerate, true_fr)
                new_ws, new_off = int(ws*framerate/true_fr), int(off*framerate/true_fr)
                rppg_ica, ppg_hr, ppg_hr_fft, ecg_hr, ecg_hr_fft, noises, sd_noises = evaluate(new_signal, ppg_row["PPG file"], ppg_row["ECG file"], new_ws, new_off, framerate, true_fr)
                n_rows, _ = rppg_ica.shape
                for i in range(n_rows):
                    row = [
                        ppg_row["Video file"], ppg_row["Tracker"], ppg_row["Region selector"], ws, off, i, framerate, 
                        hr_with_max_power(rppg_ica[i, :]), majority_vote(rppg_ica[i,:]), 
                        ppg_hr[i], ppg_hr_fft[i],
                        ecg_hr[i], ecg_hr_fft[i],
                        rppg_ica[i,0], rppg_ica[i,1], rppg_ica[i,2], noises[i, 0], sd_noises[i, 0],
                        rppg_ica[i,3], rppg_ica[i,4], rppg_ica[i,5], noises[i, 1], sd_noises[i, 1],
                        rppg_ica[i,6], rppg_ica[i,7], rppg_ica[i,8], noises[i, 2], sd_noises[i, 2]
                        ]
                    with open(sp_output, 'a') as fd:
                        writer = csv.writer(fd)
                        writer.writerow(row)
            

if __name__ == "__main__":
    files = test_data()
    ppg_meta_output = f"{PATH}output/hr_evaluation/ppg_meta_12_03_20.csv"
    sp_output = f"{PATH}output/hr_evaluation/sp_output_19_03_20.csv"
    write_ppg_out(files, ppg_meta_output)
    signal_processing_experiments(files, ppg_meta_output, sp_output)
    pass


