import pipeline
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector
from region_selection import BayesianSkinDetector, PrimitiveROI
from hr_isolator import ICAProcessor

config = Configuration(KLTBoxingWithThresholding(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 1200, 60)
t = "stat"
results = pipeline.tracking_pipeline(f"experiments/yousuf-re-run/1_{t}_2.mp4",config,display=True)
results = pipeline.tracking_pipeline(f"experiments/yousuf-re-run/1.5_{t}_2.mp4",config,display=True)
results = pipeline.tracking_pipeline(f"experiments/yousuf-re-run/2_{t}_2.mp4",config,display=True)
v, hr, x, y = results
for i in hr: 
    print(i)