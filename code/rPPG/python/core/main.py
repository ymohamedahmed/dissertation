import sys, argparse
from configuration import Configuration
from face_det import KLTBoxingWithThresholding, DNNDetector, RepeatedDetector
from region_selection import BayesianSkinDetector
from hr_isolator import ICAProcessor
from pipeline import tracking_pipeline

def main(args):
    config = Configuration(KLTBoxingWithThresholding(DNNDetector(), recompute_threshold=0.1), BayesianSkinDetector(), ICAProcessor(), 600, 60)
    # config = Configuration(RepeatedDetector(DNNDetector()), BayesianSkinDetector(), ICAProcessor(), 600, 60)
    tracking_pipeline(None, config, display=True, webcam=True)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])