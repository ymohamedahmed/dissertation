from biosppy import storage
import pandas as pd
from os import listdir
from os.path import isfile, join, isdir
from region_selection import KMeans, IntervalSkinDetector, PrimitiveROI, BayesianSkinDetector
from face_det import KLTBoxingWithThresholding, DNNDetector, RepeatedDetector
from hr_isolator import ICAProcessor, PCAProcessor, Processor
from biosppy.signals import ecg as ECG
import pyedflib
import numpy as np
from pipeline import tracking_pipeline
from configuration import Configuration, PATH
from numpy import genfromtxt
import pandas as pd
from operator import itemgetter
import time
from scipy.stats import signaltonoise

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
    cleaned = (signal-mean)<2*np.std(signal)
    return np.argmax(cleaned)

def get_ppg_signal(file_path):
    return pd.read_csv(file_path)

def upsample(data, low, high):
    x = np.arange(low, high, 1/1000)
    y = np.interp(x, data["Time"], data["PPG"])
    return y

def add_time_to_ppg(data):
    t_max, t_min = np.max(data["Timestamp"]),np.min(data["Timestamp"])
    data["Time"] = 60*(data["Timestamp"]-t_min)/(t_max-t_min)
    return data

def mean_heart_rate(signal, sampling_freq):
    time_axis, filtered, rpeaks, template_time_axis, templates, heart_rate_time_axis, heart_rate = ECG.ecg(signal=signal, sampling_rate=sampling_freq*1.0, show=False)
    avg_hr = 60*len(rpeaks)*sampling_freq/len(signal)
    return avg_hr

def track_ppg(video_path, config:Configuration):
    cap = cv.VideoCapture(PATH + video_path)
    values = []
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
                values.append(None)
                if ret == False:
                    cap.release()
                    cv.destroyAllWindows()
                    break

        frame_number += 1
        x,y,w,h = faces[0]
        
        start = time.time()
        area_of_interest, value = config.region_selector.detect(cropped)
        end = time.time()
        values.append(value)
        np.append(times, end-start)
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
        files.append([vid, None, bdf])
    return files
                
def hr_with_max_power(freqs):
    return max(freqs, key=itemgetter(1))[0]

def write_ppg_out(files):
    ppg_meta_output = f"{PATH}output/hr_evaluation/ppg_meta.csv"
    meta_columns = ["Video file", "rPPG file", "PPG file", "ECG file", "Framerate", "Tracker", "Region selector" "Time mean", "Time std", "Number of frames", "Total time", "SNR"]
    with open(ppg_meta_output, 'a') as fd:
        fd.write(meta_columns)

    for file_set in files:
        for tr in range(3):
            for rs in range(4):
                vid, ppg_file, ecg_file = file_set[0], file_set[1], file_set[1]
                config = map_config([tr, rs, 0], 0, 0)
                start = time.time()
                values, mean_time, time_std, framerate = track_ppg(vid, config)
                total = time.time()-start
                value_output = f"{vid[:-4]}-{str(config.tracker)}-{str(config.region_selector)}.csv"
                meta_row = [vid, value_output, ppg_file, ecg_file, framerate, str(config.tracker), str(config.region_selector), mean_time, time_std, len(values), total, signaltonoise(values)]
                with open(ppg_meta_output, 'a') as fd:
                    fd.write(meta_row)
                np.savetxt(value_output, values)

def evaluate(rppg_signal, ppg_file, ecg_file, window_size, offset, framerate):
    rppg_hr_pca = np.array([])
    rppg_hr_ica = np.array([])

    ecg, ecg_sf = get_ecg_signal(ecg_file)
    ecg_ws, ecg_o = len(ecg)*window_size/len(rppg_signal), len(ecg)*offset/len(rppg_signal)
    ecg_hr = []

    ppg_sf = 1000
    ppg = add_time_to_ppg(get_ppg_signal(ppg_file))
    ppg_ws, ppg_o = 60.0*window_size/len(rppg_signal), 60.0 * offset/len(rppg_signal)
    ppg_hr = []

    for i, base in enumerate(np.arange(0, len(rppg_signal)-window_size+1, offset)):
        sig = rppg_signal[base:base+window_size]
        np.append(rppg_hr_pca, PCAProcessor().get_hr(sig, framerate))
        np.append(rppg_hr_ica, ICAProcessor().get_hr(sig, framerate))

        e_low, e_high = int(i*ecg_o), int((i*ecg_o)+ecg_ws)
        ecg_hr.append(mean_heart_rate(ecg[e_low:e_high],ecg_sf))

        p_low, p_high = int(i*ppg_o), int((i*ppg_o)+ppg_ws)
        filtered = ppg[(ppg["Time"] < p_high)&(ppg["Time"]>p_low)]
        if (len(filtered) > window_size):
            signal = upsample(filtered, p_low, p_high)
            ppg_hr.append(mean_heart_rate(signal, ppg_sf))
        else: 
            ppg_hr.append(None)

    return (rppg_hr_ica, rppg_hr_pca, ppg_hr, ecg_hr)


def signal_processing_experiments(files, ppg_meta_file):
    sp_output = f"{PATH}output/hr_evaluation/sp.csv"
    columns = ["Video", "Tracker", "Region selector", "Window size", "Offset size", "Heart Rate Number", "rPPG HR ICA", "rPPG HR PCA", "PPG HR", "ECG HR", "ICA 1 HR", "ICA 1 Power", "ICA 2 HR", "ICA 2 Power", "ICA 3 HR", "ICA 3 Power", "PCA 1 HR", "PCA 1 Power", "PCA 2 HR", "PCA 2 Power", "PCA 3 HR", "PCA 3 Power"]
    ppg_meta = pd.read_csv(ppg_meta_file)
    with open(sp_output, 'a') as fd:
        fd.write(columns)
    for ppg_row in ppg_meta:
        signal = np.loadtxt(ppg_row["rPPG file"])
        for ws in np.arange(120, 1800, 10):
            for off in np.arange(10, 100, 5):
                rppg_ica, rppg_pca, ppg_hr, ecg_hr = evaluate(signal, ppg_row["PPG file"], ppg_row["ECG file"], ws, off, ppg_row["Framerate"])
                for i in range(len(rppg_ica)):
                    row = [ppg_row["Video file"], ppg_row["Tracker"], ppg_row["Region selector"], ws, off, i, hr_with_max_power(rppg_ica[i]), rppg_pca[i][0][0], ppg_hr[i], ecg_hr[i], rppg_ica[i][0][0], rppg_ica[i][0][1], rppg_ica[i][1][0], rppg_ica[i][1][1], rppg_ica[i][2][0], rppg_ica[i][2][1], rppg_pca[i][0][0], rppg_pca[i][0][1], rppg_pca[i][1][0], rppg_pca[i][1][1], rppg_pca[i][2][0], rppg_pca[i][2][1]]
                    with open(sp_output, 'a') as fd:
                        fd.write(row)
                



def map_config(config: list, window_size, offset):
    """
        Take a list of three numbers and return a configuration
        Configurations based on the following:
        tracker in {RepeatedDetector, KLTBoxingWithThresholding}
        region_selector in {PrimitiveROI, IntervalThresholding, BayesianSkinDetector(weighted=False), BayesianSkinDetector(weighted=True)}
        signal_processor in {PCA, ICA}
    """
    trackers = [RepeatedDetector(DNNDetector()), KLTBoxingWithThresholding(DNNDetector(), recompute_threshold=0.15), KLTBoxingWithThresholding(DNNDetector)]
    region_selectors = [PrimitiveROI(), IntervalSkinDetector(), BayesianSkinDetector(weighted=False), BayesianSkinDetector(weighted=True)]
    signal_processor = [PCAProcessor(), ICAProcessor()]
    t, rs, sp = config[0], config[1], config[2]
    return Configuration(trackers[t], region_selectors[rs], signal_processor[sp], window_size, offset)

files = test_data()


