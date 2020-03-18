from biosppy import storage
from biosppy.signals import ecg as ECG
import pyedflib
import numpy as np
from pipeline import tracking_pipeline
from numpy import genfromtxt

def get_ecg_signal(file_path):
    f = pyedflib.EdfReader(file_path)
    signal_labels = f.getSignalLabels()
    ecg = signal_labels.index("ECG")
    sampling_freq = f.getSampleFrequency(ecg)
    ecg = f.readSignal(ecg)[15000:]
    ecg = (ecg-np.mean(ecg))/np.std(ecg)
    f._close()
    return (ecg, sampling_freq)

def get_ppg_signal(file_path):
    return genfromtxt(file_path, delimiter=',')

def mean_heart_rate(signal, sampling_freq):
    time_axis, filtered, rpeaks, template_time_axis, templates, heart_rate_time_axis, heart_rate = ECG.ecg(signal=signal, sampling_rate=sampling_freq*1.0, show=False)
    avg_hr = 60*len(rpeaks)*sampling_freq/len(signal)
    return avg_hr

def evaluate(ppg_file, ecg_file, video, configuration):
    window_size, offset = config.window_size, config.offset
    values,pred_heart_rates,_,_,_,_,_,_,_ = tracking_pipeline(video, configuration)
    ecg, ecg_sf = get_ecg_signal(ecg_file)
    ecg_ws, ecg_o = len(ecg)*window_size/len(values), len(ecg)*offset/len(values)
    ecg_hr = mean_heart_rate(ecg,ecg_sf)

    ppg = get_ppg_signal(ppg_file)
    ppg_ws, ppg_o = len(ppg)*window_size/len(values), len(ppg)*offset/len(values)
    ppg_hr = mean_heart_rate(ppg,ppg_sf)
