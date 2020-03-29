from biosppy import storage
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

def evaluate(video, ppg_file, ecg_file, config):
    window_size, offset = config.window_size, config.offset
    values,pred_heart_rates, _, _ = tracking_pipeline(video, config)

    ecg, ecg_sf = get_ecg_signal(ecg_file)
    ecg_ws, ecg_o = len(ecg)*window_size/len(values), len(ecg)*offset/len(values)
    ecg_hr = []

    ppg_sf = 1000
    ppg = add_time_to_ppg(get_ppg_signal(ppg_file))
    # ppg_ws, ppg_o = len(ppg)*window_size/len(values), len(ppg)*offset/len(values)
    ppg_ws, ppg_o = 60.0*window_size/len(values), 60.0 * offset/len(values)
    ppg_hr = []

    for i in range(len(pred_heart_rates)):
        e_low, e_high = int(i*ecg_o), int((i*ecg_o)+ecg_ws)
        ecg_hr.append(mean_heart_rate(ecg[e_low:e_high],ecg_sf))

        #TODO add check for missing ppg data
        p_low, p_high = int(i*ppg_o), int((i*ppg_o)+ppg_ws)
        filtered = ppg[(ppg["Time"] < p_high)&(ppg["Time"]>p_low)]
        if (len(filtered) > window_size):
            signal = upsample(filtered, p_low, p_high)
            ppg_hr.append(mean_heart_rate(signal, ppg_sf))
        else: 
            ppg_hr.append(None)

    return (pred_heart_rates, ppg_hr, ecg_hr)

def track_ppg(video_path, config:Configuration):
    cap = cv.VideoCapture(PATH + video_path)
    values = []
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_number = 0
    frame_rate = int(cap.get(cv.CAP_PROP_FPS))

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
        
        area_of_interest, value = config.region_selector.detect(cropped)
        values.append(value)
            
        start = Timing.time()
        return values

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
    ppp_meta_output = f"{PATH}output/hr_evaluation/ppg_meta.csv"
    meta_columns = ["Video file", "PPG file", "Tracker", "Region selector" "Time mean", "Time std", "Number of frames", "Total time", "SNR"]
    with open(ppg_meta_output, 'a') as fd:
        fd.write(meta_columns)

    for vid in files[:, 0]:
        for tr in range(3):
            for rs in range(4):
                config = map_config([tr, rs, 0], 0, 0)
                start = time.time()
                values, mean_time, time_std = track_ppg(vid, config)
                total = time.time()-start
                value_output = f"{vid[:-4]}-{str(config.tracker)}-{str(config.region_selector)}.csv"
                meta_row = [vid, value_output, str(config.tracker), str(config.region_selector), mean_time, time_std, len(values), total, signaltonoise(values)]
                with open(ppg_meta_output, 'a') as fd:
                    fd.write(meta_row)
                np.savetxt(value_output, values)

def signal_processing_experiments(files):
    sp_output = f"{PATH}output/hr_evaluation/sp.csv"
    rows = ["Video", "Distance", "Exercise", "Detector", "ROI", "Signal processor", "Window size", "Offset size", "Repeat", "Heart Rate Number", "rPPG HR", "PPG HR", "ECG HR", "SNR of rPPG", "ICA 1 HR", "ICA 1 Power", "ICA 2 HR", "ICA 2 Power", "ICA 3 HR", "ICA 3 Power", "PCA 1 HR", "PCA 1 Power", "PCA 2 HR", "PCA 2 Power", "PCA 3 HR", "PCA 3 Power"]


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
config = Configuration(KLTBoxingWithThresholding(DNNDetector(), 0.2), PrimitiveROI(), ICAProcessor(), 1200, 60)
output_path = "rPPG/output/hr_evaluation.csv"
rows = ["Video", "Distance", "Exercise", "Detector", "ROI", "Signal processor", "Window size", "Offset size", "Repeat", "Heart Rate Number", "rPPG HR", "PPG HR", "ECG HR", "SNR of rPPG", "ICA 1 HR", "ICA 1 Power", "ICA 2 HR", "ICA 2 Power", "ICA 3 HR", "ICA 3 Power", "PCA 1 HR", "PCA 1 Power", "PCA 2 HR", "PCA 2 Power", "PCA 3 HR", "PCA 3 Power"]

for file in files:
    results = evaluate(file[0], file[1], file[2], config)
    for i in range(len(results[0])):
        powers = results[0][i]
        ica = [powers[0], powers[]] if ica else [None for _ in range(6)]
        pca = []
        row = [file[0], "", "", str(config.tracker), str(config.region_selector), str(config.signal_processor), window_size, offset_size, "", i, ]
"""
            print(f"Considering base file: {file}")
            result = evaluate(f"{PATH}{file}.csv", f"{PATH}{file}.EDF", f"{PATH}{file}.mp4", config)
            print(result)
            print(f"ECG: {result[0]}")
            print(f"rPPG: {result[1]}")
            print(f"PPG: {result[2]}")
            break


"""