import sklearn
import glob
import cv2 as cv
from sklearn.cluster import KMeans
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
import scipy
import tikzplotlib
import pyedflib
import time as Timing
from configuration import PATH

class FaceTracker(object):
    def __init__(self, detector, scaled_width=300, scaled_height=300):
        self.detector = detector
        self.frame_number = 0
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height

    def track(self, frame):
        scaled_frame = self._scale_down_fixed_size(frame)
        faces, profiling = self._track(scaled_frame)
        self.frame_number += 1
        height, width, _  = frame.shape
        faces = [self._bound_coordinates(f, self.scaled_width,self.scaled_height) for f in faces]
        faces = [self._scale_face(frame, f) for f in faces]
        return [self._bound_coordinates(f, width, height) for f in faces], frame, (self._crop_image(frame, *(faces[0])) if len(faces) > 0 else frame), profiling

    def overlay(self, frame, faces):
        self._draw_rectangle(frame, faces)
        self._overlay(frame, self.scaled_width, self.scaled_height)

    def _detect(self, frame):
        faces =  self.detector.detect_face(frame)
        return faces

    def _draw_rectangle(self, image, faces):
        for (x, y, w, h) in faces:
            x,y,w,h = int(x),int(y),int(w),int(h)
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return image

    @staticmethod
    def _scale_down(image, scale):
        return cv.resize(image, (int(image.shape[1]//scale),int(image.shape[0]//scale)), interpolation = cv.INTER_AREA)

    def _scale_down_fixed_size(self, image):
        return cv.resize(image, (self.scaled_width, self.scaled_height), interpolation = cv.INTER_AREA)

    def _scale_face(self, image, face):
        x,y,w,h = face
        width = int(w * (image.shape[1]/self.scaled_width))
        height = int(h * (image.shape[0]/self.scaled_height))
        x2 = int(x/self.scaled_width * image.shape[1])
        y2 = int(y/self.scaled_height * image.shape[0])
        return (x2,y2,width,height)

    def _bound_coordinates(self, face, width, height):
        x,y,w,h = face
        x1 = 0 if x < 0 else width - 1 if x > width else x
        y1 = 0 if y < 0 else height - 1 if y > height else y
        x2 = 0 if x+w < 0 else width - 1 if x+w > width else x+w
        y2 = 0 if y+h < 0 else height - 1 if y+h > height else y+h
        return int(x1), int(y1), int(x2-x1), int(y2-y1)

    def _crop_image(self, image, x, y, w, h):
        """Coordinates should have been bound to fall within the image"""
        x1,y1,x2,y2 = x,y,w+x,h+y
        return image[y1:y2, x1:x2]

class NaiveKLTBoxing(FaceTracker):

    def __init__(self, detector):
        self.feature_params = dict( maxCorners = 5000,
                              qualityLevel = 0.01,
                              minDistance = 1,
                              blockSize = 7 )
        self.lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) 
        self.old_points = None
        super().__init__(detector)

    def _overlay(self, frame, scaled_width, scaled_height):
        h,w,_ = frame.shape
        for i in self.old_points:
          x,y = i.ravel()
          x = int(w*x/scaled_width)
          y = int(h*y/scaled_height)
          cv.circle(frame,(x,y),3,255,-1)
  
    def _to_gray(self, frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    def _features_to_track(self, frame):
        faces = self._detect(frame)
        time_to_select_points = None
        d = np.array([])
        self.old_frame = frame
        if(len(faces) > 0):
            face_found = True
            height, width, _ = frame.shape
            x,y,w,h = self._bound_coordinates(faces[0], width, height)
            # padding works as ((top, bottom), (left, right))
            face_mask = np.pad(np.ones(shape=(h,w)), ((y, height-y-h),(x, width-x-w)) , 'constant', constant_values=0)
            start = Timing.time()
            self.old_points = cv.goodFeaturesToTrack(self._to_gray(frame),mask=np.uint8(face_mask), **self.feature_params)
            self.original_points  = self.old_points
            time_to_select_points = Timing.time()-start
        return faces, time_to_select_points
  
    def _track_points(self, frame):
        start = Timing.time()
        new_points, st, err = cv.calcOpticalFlowPyrLK(self._to_gray(self.old_frame), self._to_gray(frame), self.old_points, None, **self.lk_params)
        time_to_track_points = Timing.time() - start
        
        new_points = new_points[st==1]
        self.old_frame = frame
        height, width, _ = frame.shape
        min_x,min_y, max_x, max_y = width, height, 0, 0
        for i in new_points:
            x,y = i.ravel()
            min_x, min_y = min(x, min_x), min(y, min_y)
            max_x, max_y = max(x, max_x), max(y, max_y)
        x,y,w,h = min_x,min_y,max_x-min_x, max_y-min_y
        faces = [(x,y,w,h)]
        d = np.sqrt(np.sum(np.square(np.abs(new_points-self.old_points)), axis=1))
        orig_d = np.sqrt(np.sum(np.square(np.abs(new_points-self.original_points)), axis=1))
        self.old_points = new_points.reshape(-1,1,2)
        return faces, time_to_track_points, np.mean(d), np.std(d), np.mean(orig_d), np.std(orig_d)

    def _track(self, frame):
        if self.frame_number == 0:
            f, t = self._features_to_track(frame)
            return f, {}
        else:
            f, t = self._track_points(frame)
            return f, {}

    def __str__(self):
        return f"{self.__class__.__name__}-{str(self.detector)}-scale_{self.scale}"

class KLTBoxingWithThresholding(NaiveKLTBoxing):

    def __init__(self, detector, recompute_threshold = 0.25):
        super().__init__(detector)
        self.detector = detector
        self.recompute_threshold = recompute_threshold
        self.old_frame = None
        self.old_points = None
        self.cumulative_change = 0
        self.redetects = 0

    def _redetect(self, frame):
        faces, time = self._features_to_track(frame)
        self.cumulative_change = 0
        self.redetects += 1
        if(len(faces) > 0):
            _,_,w,h = faces[0]
            self.estimated_size = w*h
        return faces, time
        
    def _track(self, frame):
        faces = None
        profiling = {}
        if self.frame_number == 0 or self.cumulative_change > self.recompute_threshold:
            faces, time = self._redetect(frame)
            profiling["time_to_select_points"] = time
        else: 
            faces, time, mean_d, std_d, mean_orig_d, std_orig_d = self._track_points(frame)
            profiling["time_to_track_points"] = time
            profiling["point_distance_mean"] = mean_d
            profiling["point_distance_std"] = std_d
            profiling["orig_point_distance_mean"] = mean_orig_d
            profiling["orig_point_distance_std"] = std_orig_d

            x,y,w,h = faces[0]
            increase_in_size = abs(self.estimated_size - (w*h))/self.estimated_size
            self.cumulative_change += increase_in_size + (increase_in_size*self.cumulative_change)
            profiling["cumulative_change"] = self.cumulative_change
            self.estimated_size = w*h
            if self.cumulative_change > self.recompute_threshold:
                faces, time = self._redetect(frame)
                profiling["time_to_select_points"] = time
        return faces, profiling
  
    def __del__(self):
        if(self.frame_number > 0):
            print(f"{self.redetects}/{self.frame_number}={100*self.redetects/self.frame_number}%")

    def __str__(self):
        return f"{self.__class__.__name__}-{str(self.detector)}"

class RepeatedDetector(FaceTracker):

  def __init__(self, detector):
    super().__init__(detector=detector)
    self.detector = detector

  def _track(self, frame):
    return self._detect(frame),{}
  
  def _overlay(self, frame, scaled_width, scaled_height):
    pass

  def __str__(self):
     return f"{self.__class__.__name__}-{str(self.detector)}"

class DNNDetector():
    def __init__(self):
        MODEL_PATH = f"{PATH}models/"
        modelFile = MODEL_PATH + "opencv_face_detector_uint8.pb"
        configFile = MODEL_PATH + "opencv_face_detector.pbtxt"
        self.dnn = cv.dnn.readNetFromTensorflow(modelFile, configFile)

    def detect_face(self, image):
        faces = []
        # print("FACE DETECTION")
        blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        h,w,_ = image.shape
        self.dnn.setInput(blob)
        detections = self.dnn.forward()
        # print(f"Number of detections: {len(detections)}")
        # for i in range(detections.shape[2]):
        #     confidence = detections[0, 0, i, 2]
        #     print(f"Confidence: {confidence}")
        #     if confidence > 0.5:
        #         x1 = int(detections[0, 0, i, 3] * w)
        #         y1 = int(detections[0, 0, i, 4] * h)
        #         x2 = int(detections[0, 0, i, 5] * w)
        #         y2 = int(detections[0, 0, i, 6] * h)
        #         faces.append((x1,y1,x2-x1,y2-y1))
        if detections.shape[2] > 0 and detections[0,0,0,2] > 0.3:
            x1 = int(detections[0, 0, 0, 3] * w)
            y1 = int(detections[0, 0, 0, 4] * h)
            x2 = int(detections[0, 0, 0, 5] * w)
            y2 = int(detections[0, 0, 0, 6] * h)
            return [(x1,y1,x2-x1,y2-y1)]
        else:
            return []
    
    def __str__(self):
        return self.__class__.__name__