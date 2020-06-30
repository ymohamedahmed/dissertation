from biosppy import storage
from biosppy.signals import ecg as ECG
import pyedflib
import numpy as np
from os import listdir
from os.path import isfile, join

def heart_rates_bdf(file_name):
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    ecg = signal_labels.index("EXG1")
    sampling_freq = f.getSampleFrequency(ecg)
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    ecg = f.readSignal(ecg)[1000:]
    f._close()
    time_axis, filtered, rpeaks, template_time_axis, templates, heart_rate_time_axis, heart_rate = ECG.ecg(signal=ecg, sampling_rate=sampling_freq*1.0, show=False)
    avg_hr = 60*len(rpeaks)*sampling_freq/len(ecg)
    return avg_hr
    
def get_avi_bdf(root, folder_name):
    path = f"{root}/mahnob/{folder_name}"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))] 
    avi = [f for f in onlyfiles if f.endswith(".avi")]
    bdf = [f for f in onlyfiles if f.endswith(".bdf")]
    return avi[0], bdf[0]