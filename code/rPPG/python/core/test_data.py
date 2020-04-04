import pipeline
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector
from region_selection import BayesianSkinDetector, PrimitiveROI
from hr_isolator import ICAProcessor

config = Configuration(KLTBoxingWithThresholding(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 1200, 60)
results = pipeline.tracking_pipeline("experiments/yousuf-re-run/1_jog_1.mp4",config,display=True)
v, hr, x, y = results
for i in hr: 
    print(i)