import time as Timing
import numpy as np
import cv2 as cv
from face_det import KLTBoxingWithThresholding, FaceTracker, RepeatedDetector, DNNDetector
import pandas as pd
from mahnob import get_avi_bdf
from configuration import PATH

def matrix_from_face(face, width, height):
    x,y,w,h = face
    # print(f"Frame width: {width}, frame height: {height}, face width: {w}, face height: {h}, x pos: {x}, y pos: {y}")
    matrix = np.ones(shape=(h,w), dtype=np.bool)
    matrix = np.pad(matrix, ((y,height-(y+h)),(x,width-(x+w))), 'constant', constant_values=0)
    matrix = np.repeat(matrix[:, :, np.newaxis], 3, axis=2)
    return matrix
    cls = ["Video", "Stationary", "Threshold", "Frame number", "Time of face tracker", "Time of face detector", "FN", "FP", "TN", "TP", "Time to select points", "Time to track points"]

def evaluation_pipeline(detector, tracker, video_path):
    # Profiling
    total_start = Timing.time()
    results = []
    
    
    cap = cv.VideoCapture(PATH + video_path)
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
#         print(f"FN: {fn}, FP: {fp}, TN: {tn}, TP:{tp}")
        # Format (video name, frame number, time_to_track, time_to_detect, FN, FP, TN, TP, time_to_select_points, time_to_track_points)
        selecting = profiling_tr["time_to_select_points"] if "time_to_select_points" in profiling_tr else None
        tracking = (profiling_tr["time_to_track_points"], profiling_tr["point_distance_mean"], profiling_tr["point_distance_std"]) if "time_to_track_points" in profiling_tr else (None, None, None)
        results.append([video_path, tracker.recompute_threshold, frame_number, time_tr, time_det, fn, fp, tn, tp, selecting, tracking[0], tracking[1], tracking[2]])
    return results
#     return (values, heart_rates, frame_number, total_time, time_tracking, time_roi, time_display, time_ica, time_read)
def tracker_vs_detector(video, threshold):
    video_path = f"{video}"
    video_name = video.split(".")[0]
    results = evaluation_pipeline(RepeatedDetector(DNNDetector()), KLTBoxingWithThresholding(DNNDetector(), recompute_threshold=threshold), video_path)
    start = Timing.time()
    overall_results = pd.DataFrame(data=results, columns=cls)
    print(f"Time to dataframe: {Timing.time()-start}")
#     overall_results = overall_results.append(results)
#     overall_results.to_csv(f"{PATH}output/tracking_vs_detecting_{video_name}.csv")
    return overall_results

folders = ["21", "22", "23", "24", "25", "26", "27", "28", "29"]
videos = []
movement_vids = ["mov-1.mp4", "mov-2.mp4", "mov-3.mp4"]
stationary_vids = []
movement_vids = [f"test-face-detection-videos/{v}" for v in movement_vids]
for folder in folders:
    avi, _ = get_avi_bdf(PATH, folder)
    stationary_vids.append(f"mahnob/{folder}/{avi}")
videos = movement_vids + stationary_vids
np.random.shuffle(videos)
print(videos)
cls = ["Video", "Threshold", "Frame number", "Time of face tracker", "Time of face detector", "FN", "FP", "TN", "TP", "Time to select points", "Time to track points", "Point distance mean", "Point distance std."]
"""
results = pd.DataFrame(columns=cls)
results = results.append(tracker_vs_detector("test-face-detection-videos/mov-3.mp4", 0.4))
results.to_csv(f"{PATH}output/mov_3_tweaked_scaling.csv")

"""
# thresholds = [0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]
# thresholds = [0.45, 0.5, 0.55, 0.6]
# thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2]
# thresholds = [0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.325, 0.35, 0.375, 0.4]
# results = tracker_vs_detector("test-face-detection-videos/mov-1.mp4",0.15)
# results.to_csv(f"{PATH}/mov-1-error-correction.csv")
thresholds = [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
results = pd.DataFrame(columns=cls)
for t_index, t in enumerate(thresholds): 
    for v_index, v in enumerate(videos):
        print(f"Beginning experiment: {(t_index*len(videos))+v_index}/{len(thresholds)*len(videos)} threshold: {t} and video: {v}")
        results = results.append(tracker_vs_detector(v,t))
        results.to_csv(f"{PATH}output/tracking_vs_detecting_large_scale_6.csv")
