import pipeline
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector
from region_selection import BayesianSkinDetector
from hr_isolator import ICAProcessor

config = Configuration(KLTBoxingWithThresholding(DNNDetector()),  BayesianSkinDetector(), ICAProcessor(), 1200, 60)
results = pipeline.tracking_pipeline("mahnob/21/P1-Rec1-2009.07.09.17.53.46_C1 trigger _C_Section_21.avi",config,display=True)