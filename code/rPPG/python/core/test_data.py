import pipeline
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector
from region_selection import BayesianSkinDetector
from hr_isolator import ICAProcessor

config = Configuration(KLTBoxingWithThresholding(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 1200, 60)
results = pipeline.tracking_pipeline("experiments/yousuf-re-run/2_star_1.mp4",config,display=True)