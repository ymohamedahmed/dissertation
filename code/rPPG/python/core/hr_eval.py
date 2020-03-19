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
    return genfromtxt(file_path, delimiter=',')

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

    ppg_sf = 64.0
    ppg = get_ppg_signal(ppg_file)
    ppg_ws, ppg_o = len(ppg)*window_size/len(values), len(ppg)*offset/len(values)
    ppg_hr = []

    for i in range(len(pred_heart_rates)):
        ecg_hr.append(mean_heart_rate(ecg[i*ecg_o:(i*ecg_o)+ecg_ws],ecg_sf))
        ppg_hr.append(mean_heart_rate(ppg[i*ppg_o:(i*ppg_o)+ppg_ws],ppg_sf))

    return (ecg_hr, pred_heart_rates, ppg_hr)   

config = Configuration(KLTBoxingWithThresholding(DNNDetector(), 0.2), PrimitiveROI(), ICAProcessor(), np.mean, 1200, 60)
# evaluate()

