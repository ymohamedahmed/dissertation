import sys, argparse
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector
from region_selection import BayesianSkinDetector
from hr_isolator import ICAProcessor
from pipeline import tracking_pipeline

def main(args):
    config = Configuration(KLTBoxingWithThresholding(DNNDetector()), BayesianSkinDetector(), ICAProcessor(), 600, 60)
    tracking_pipeline(None, config, display=True, webcam=True)

if __name__ == "__main__":
    main(sys.argv[1:])