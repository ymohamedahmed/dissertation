import pipeline
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector, RepeatedDetector
from region_selection import BayesianSkinDetector, PrimitiveROI, IntervalSkinDetector
from hr_isolator import ICAProcessor
import cv2 as cv

# video = "mahnob/Sessions/1042/P9-Rec1-2009.07.23.14.55.41_C1 trigger _C_Section_2.avi"
# video = "experiments/candidate-2-yousuf/1_stat.mp4"
for v in ["1-cut.mp4", "2-cut.avi", "3-cut.avi", "4-cut.avi"]:
    video = f"skin_tone_videos/{v}"
    config = Configuration(KLTBoxingWithThresholding(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 600, 60)
    pipeline.tracking_pipeline(video, config, display=True)