from biosppy import storage
from biosppy.signals import ecg as ECG
import pyedflib
import numpy as np


def get_signal(file_name):
    f = pyedflib.EdfReader(file_name)
    signal_labels = f.getSignalLabels()
    ecg = signal_labels.index("ECG")
    sampling_freq = f.getSampleFrequency(ecg)
    ecg = f.readSignal(ecg)[15000:]
    ecg = (ecg-np.mean(ecg))/np.std(ecg)
    f._close()
    return (ecg, sampling_freq)

def mean_heart_rate(signal, sampling_freq):
    time_axis, filtered, rpeaks, template_time_axis, templates, heart_rate_time_axis, heart_rate = ECG.ecg(signal=signal, sampling_rate=sampling_freq*1.0, show=False)
    avg_hr = 60*len(rpeaks)*sampling_freq/len(signal)
    return avg_hr

def evaluate(ppg_file, ecg_file, video, configuration):
    tracking_pipeline()

def evaluate_hr(rppg_hr, ecg_hr, number_of_frames, window_size, step_size):
    correct = []
    for index, hr in enumerate(rppg_hr):
        progress = window_size + (index*step_size)
        progress = int(len(ecg_hr)*progress/number_of_frames)
        start = int((len(ecg_hr)*index*step_size)/number_of_frames)
        reference = ecg_hr[start:progress]
        correct_ref = np.min(reference) <= hr and np.max(reference) >= hr
        correct.append(correct_ref)
    return correct




