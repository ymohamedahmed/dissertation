import face_det
from face_det import FaceTracker
from region_selection import RegionSelector
from hr_isolator import Processor

class Configuration():
    def __init__(self, tracker: FaceTracker,  region_selector: RegionSelector, signal_processor: Processor, aggregate_function, window_size:int, offset:int):
        self.tracker = tracker
        self.region_selector = region_selector
        self.signal_processor = signal_processor
        self.aggregate_function = aggregate_function
        self.window_size = window_size
        self.offset = offset