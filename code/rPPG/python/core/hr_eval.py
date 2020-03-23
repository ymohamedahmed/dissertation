from biosppy import storage
from region_selection import KMeans, IntervalSkinDetector, PrimitiveROI, BayesianSkinDetector
from face_det import KLTBoxingWithThresholding, DNNDetector
from hr_isolator import ICAProcessor, PCAProcessor, Processor
from biosppy.signals import ecg as ECG
import pyedflib
import numpy as np
from pipeline import tracking_pipeline
from configuration import Configuration
from numpy import genfromtxt
import pandas as pd

def start_ecg_point(signal):
    mean = np.mean(signal)
    cleaned = (signal-mean)<2*np.std(signal)
    return np.argmax(cleaned)

def get_ecg_signal(file_path):
    f = pyedflib.EdfReader(file_path)
    signal_labels = f.getSignalLabels()
    ecg = signal_labels.index("ECG")
    sampling_freq = f.getSampleFrequency(ecg)
    ecg = f.readSignal(ecg)
    ecg = ecg[start_ecg_point(ecg):]
    ecg = (ecg-np.mean(ecg))/np.std(ecg)
    f._close()
    return (ecg, sampling_freq)

def get_ppg_signal(file_path):
    return pd.read_csv(file_path)

def upsample(data):
    t_max, t_min = np.max(data["Timestamp"]),np.min(data["Timestamp"])
    ppg["Time"] = 60*(data["Timestamp"]-t_min)/(t_max-t_min)
    x = np.arange(0, 60, 1/1000)
    y = np.interp(x, data["Time"], data["PPG"])
    return y

def mean_heart_rate(signal, sampling_freq):
    time_axis, filtered, rpeaks, template_time_axis, templates, heart_rate_time_axis, heart_rate = ECG.ecg(signal=signal, sampling_rate=sampling_freq*1.0, show=False)
    avg_hr = 60*len(rpeaks)*sampling_freq/len(signal)
    return avg_hr

def evaluate(ppg_file, ecg_file, video, config):
    window_size, offset = config.window_size, config.offset
    values,pred_heart_rates, _, _ = tracking_pipeline(video, config)

    ecg, ecg_sf = get_ecg_signal(ecg_file)
    ecg_ws, ecg_o = len(ecg)*window_size/len(values), len(ecg)*offset/len(values)
    ecg_hr = []

    ppg_sf = 1000
    ppg = upsample(get_ppg_signal(ppg_file))
    ppg_ws, ppg_o = len(ppg)*window_size/len(values), len(ppg)*offset/len(values)
    ppg_hr = []

    for i in range(len(pred_heart_rates)):
        e_low, e_high = int(i*ecg_o), int((i*ecg_o)+ecg_ws)
        ecg_hr.append(mean_heart_rate(ecg[e_low:e_high],ecg_sf))
        p_low, p_high = int(i*ppg_o), int((i*ppg_o)+ppg_ws)
        ppg_hr.append(mean_heart_rate(ppg[p_low:p_high],ppg_sf))

    return (ecg_hr, pred_heart_rates, ppg_hr)   

def mean(image):
    return image.mean(axis=0).mean(axis=0)

config = Configuration(KLTBoxingWithThresholding(DNNDetector(), 0.2), PrimitiveROI(), ICAProcessor(), mean, 1200, 60)
base_path = "experiments/yousuf-re-run/"
output_path = "rPPG/output/hr_evaluation.csv"
rows = ["Video", "Distance", "Exercise", "Detector", "ROI", "Signal processor", "Window size", "Offset size", "Repeat", "Heart Rate Number", "ECG HR", "PPG HR", "rPPG HR", "SNR of rPPG"]
for dist in ["1", "1.5", "2"]:
    for exer in ["stat", "star", "jog"]:
        for repeat in range(1,4):
            file = f"{base_path}{dist}_{exer}_{repeat}"
            print(f"Considering base file: {file}")
            result = evaluate(f"rPPG/{file}.csv", f"rPPG/{file}.EDF", f"{file}.mp4", config)
            print(result)
            print(f"ECG: {result[0]}")
            print(f"rPPG: {result[1]}")
            print(f"PPG: {result[2]}")
            break


