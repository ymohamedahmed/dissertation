import pipeline
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector, RepeatedDetector
from region_selection import BayesianSkinDetector, PrimitiveROI
from hr_isolator import ICAProcessor
import cv2 as cv

config = Configuration(KLTBoxingWithThresholding(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 600, 60)
# config = Configuration(RepeatedDetector(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 1200, 60)
t = "stat"
# results = pipeline.tracking_pipeline(f"experiments/yousuf-re-run/1_{t}_2.mp4",config,display=True)
# results = pipeline.tracking_pipeline(f"experiments/yousuf-re-run/1.5_{t}_2.mp4",config,display=True)
# results = pipeline.tracking_pipeline(f"experiments/yousuf-re-run/2_{t}_2.mp4",config,display=True)
pipeline.tracking_pipeline("", config, display=True, webcam=True)
cv.destroyAllWindows()