PATH = "/Users/yousuf/Workspace/dissertation/code/rPPG/"

class Configuration():
    def __init__(self, tracker, region_selector, signal_processor, window_size:int, offset:int):
        self.tracker = tracker
        self.region_selector = region_selector
        self.signal_processor = signal_processor
        self.window_size = window_size
        self.offset = offset