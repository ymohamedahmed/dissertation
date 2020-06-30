import pipeline
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector, RepeatedDetector
from region_selection import BayesianSkinDetector, PrimitiveROI
from hr_isolator import ICAProcessor
import cv2 as cv

config = Configuration(KLTBoxingWithThresholding(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 600, 60)
t = "stat"
pipeline.tracking_pipeline("", config, display=True, webcam=True)
cv.destroyAllWindows()