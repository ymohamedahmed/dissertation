PATH = "/home/yousuf/workspace/dissertation/code/rPPG/"
class Configuration():
    def __init__(self, tracker,  region_selector, signal_processor, aggregate_function, window_size:int, offset:int):
        self.tracker = tracker
        self.region_selector = region_selector
        self.signal_processor = signal_processor
        self.aggregate_function = aggregate_function
        self.window_size = window_size
        self.offset = offset