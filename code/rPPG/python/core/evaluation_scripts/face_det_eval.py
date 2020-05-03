import time as Timing
import numpy as np
import cv2 as cv
from face_det import KLTBoxingWithThresholding, FaceTracker, RepeatedDetector, DNNDetector
import pandas as pd
from mahnob import get_avi_bdf
from configuration import PATH

def matrix_from_face(face, width, height):
    x,y,w,h = face
    matrix = np.ones(shape=(h,w), dtype=np.bool)
    matrix = np.pad(matrix, ((y,height-(y+h)),(x,width-(x+w))), 'constant', constant_values=0)
    matrix = np.repeat(matrix[:, :, np.newaxis], 3, axis=2)
    return matrix

def evaluation_pipeline(detector, tracker, video_path):
    # Profiling
    total_start = Timing.time()
    results = []
    cap = cv.VideoCapture(video_path)
    heart_rates = []
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
        faces_tr, faces_det = [],[]
        time_tr, time_det = 0,0
        while(not(face_found)):
            start = Timing.time()
            faces_tr, frame_tr, cropped_tr, profiling_tr = tracker.track(frame)
            time_tr = (Timing.time() - start)
            
            start = Timing.time()
            faces_det, frame_det, cropped_det, profiling_det = detector.track(frame)
            time_det = (Timing.time() - start)
            
            if(len(faces_tr) > 0 and len(faces_det) > 0):
                face_found = True
            else:
                ret, frame = cap.read()
                if ret == False:
                    cap.release()
                    cv.destroyAllWindows()
                    break
        frame_number += 1
        T = matrix_from_face(faces_tr[0], width, height)
        D = matrix_from_face(faces_det[0], width, height)
        fn = np.sum(np.bitwise_and(np.invert(T), D))
        fp = np.sum(np.bitwise_and(T,np.invert(D)))
        tn = np.sum(np.bitwise_and(np.invert(T),np.invert(D)))
        tp = np.sum(np.bitwise_and(T,D))
        selecting = profiling_tr["time_to_select_points"] if "time_to_select_points" in profiling_tr else None
        tracking = (profiling_tr["time_to_track_points"], profiling_tr["point_distance_mean"], profiling_tr["point_distance_std"], profiling_tr["orig_point_distance_mean"], profiling_tr["orig_point_distance_std"], profiling_tr["cumulative_change"]) if "time_to_track_points" in profiling_tr else (None, None, None, None, None, None)
        results.append([video_path, tracker.recompute_threshold, frame_number, time_tr, time_det, fn, fp, tn, tp, selecting, tracking[0], tracking[1], tracking[2], tracking[3], tracking[4], tracking[5]])
    return results

def tracker_vs_detector(video, threshold):
    video_path = f"{video}"
    video_name = video.split(".")[0]
    results = evaluation_pipeline(RepeatedDetector(DNNDetector()), KLTBoxingWithThresholding(DNNDetector(), recompute_threshold=threshold), video_path)
    start = Timing.time()
    overall_results = pd.DataFrame(data=results, columns=cls)
    print(f"Time to dataframe: {Timing.time()-start}")
    return overall_results

videos = []
for d in [1, 1.5, 2]:
    for e in ["stat", "star", "jog"]:
        for r in [1,2,3]:
            videos.append(f"{PATH}experiments/yousuf-re-run/{d}_{e}_{r}.mp4")
print(videos)
cls = ["Video", "Threshold", "Frame number", "Time of face tracker", "Time of face detector", "FN", "FP", "TN", "TP", "Time to select points", "Time to track points", "Point distance mean", "Point distance std.", "Point distance original mean.", "Point distance original std.", "Cumulative change"]

thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.35, 0.45]
results = pd.DataFrame(columns=cls)
for t_index, t in enumerate(thresholds): 
    for v_index, v in enumerate(videos):
        print(f"Beginning experiment: {(t_index*len(videos))+v_index}/{len(thresholds)*len(videos)} threshold: {t} and video: {v}")
        results = results.append(tracker_vs_detector(v,t))
        results.to_csv(f"{PATH}output/tracking_vs_detecting_13_04_20.csv")
