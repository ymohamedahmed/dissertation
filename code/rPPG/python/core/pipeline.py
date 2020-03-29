import time as Timing
import numpy as np
import cv2 as cv
from visualisation import Visualiser
from configuration import Configuration, PATH

def tracking_pipeline(video_path, config:Configuration, display = False):
    # Profiling
    total_start = Timing.time()
    time_read = 0
    time_tracking = 0
    time_roi = 0
    time_display = 0
    time_ica = 0
    
    cap = cv.VideoCapture(PATH + video_path)
    values = []
    heart_rates = []
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_number = 0
    frame_rate = int(cap.get(cv.CAP_PROP_FPS))
    visualiser = None
    if display:
        visualiser = Visualiser(PATH, config.tracker, video_path, config.region_selector, config.signal_processor, cap)
    while(cap.isOpened()):
        start = Timing.time()
        ret, frame = cap.read()
        time_read += (Timing.time()-start)
        
        if ret == False:
            cap.release()
            cv.destroyAllWindows()
            break

        face_found = False
        faces, cropped = None, None
        while(not(face_found)):
            start = Timing.time()
            faces, frame, cropped, profiling = config.tracker.track(frame)
            time_tracking += (Timing.time() - start)
            
            
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
        
        start = Timing.time()
        try: 
            area_of_interest, value = config.region_selector.detect(cropped)
        except Exception as e: 
            print("ERROR")
            print(e)
            print(f"FACE: ({x},{y},{w},{h})")
            print(f"Frame shape: {frame.shape}")
            print(f"Cropped: {cropped.shape}")
            print(frame)
            print(frame.shape)
        time_roi += (Timing.time()-start)
        
    # area_of_interest = np.pad(area_of_interest, ((y,height-(y+h)),(x,width-(x+w))), 'constant', constant_values=1)
    # area_of_interest = np.repeat(area_of_interest[:, :, np.newaxis], 3, axis=2)
    # frame = np.ma.masked_array(frame, mask=area_of_interest)
    # value = config.aggregate_function(frame)
        values.append(value)
        if display:
            start = Timing.time()
            visualiser.display(frame, faces, area_of_interest)
            time_display += (Timing.time()-start)
        
        start = Timing.time()
        # if this is the first time we've reached the required number of values
        if frame_number % config.window_size == 0 and frame_number//config.window_size == 1:
            print(np.array(values[:config.window_size]).shape)
            print(np.array(values[:config.window_size]))
            heart_rates.append(config.signal_processor.get_hr(np.array(values[:config.window_size]), frame_rate))
        elif (frame_number-config.window_size)%config.offset == 0 and frame_number>config.window_size:
            n = int((frame_number-config.window_size)/config.offset)
            print(f"N: {n}, Frame number: {frame_number}")
            heart_rates.append(config.signal_processor.get_hr(np.array(values[n*config.offset : (n*config.offset)+config.window_size]), frame_rate))
        time_ica += (Timing.time()-start)

    print(f"Total times: tracking {time_tracking}s, ROI {time_roi}s, display {time_display}s, ICA {time_ica}s, read {time_read}")
    print(f"Average per frame: tracking {time_tracking/frame_number}s, ROI {time_roi/frame_number}s, display {time_display/frame_number}s, ICA {time_ica/frame_number}s, reading video {time_read/frame_number}")
    print(f"Frames per second: tracking {frame_number/time_tracking}, ROI {frame_number/time_roi}, display {frame_number/time_display if display else 0}, ICA {frame_number/time_ica}, reading video {frame_number/time_read}")
    print(f"Number of frames: {frame_number}")
    total_time = Timing.time() - total_start
    return (values, heart_rates, frame_number, frame_rate)