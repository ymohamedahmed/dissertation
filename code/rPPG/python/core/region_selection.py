import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.stats import truncnorm

def _cluster_skin_distance(ycrcb_colour):
    """
        We define skin threshold as (137, 74) => (181, 126) and compute distance from a point to this rectangle
    """
    x,y = (137+181)/2.0, (74+126)/2.0
    px,py = ycrcb_colour[1], ycrcb_colour[2]
    dx = max(abs(px - x) - width / 2, 0)
    dy = max(abs(py - y) - height / 2, 0)
    return (dx * dx) + (dy * dy)

def skin_tone(frame, alpha=3, beta=1, clusters=5):
    image = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    image[:, :, 0] = 0
    h,w,_ = image.shape
    arr = image.reshape((h*w,3))
    kmeans = KMeans(n_clusters=clusters, n_jobs=-1, max_iter=50).fit(arr)
    centers = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    scores = [(alpha*counts[i]) - (beta*_cluster_skin_distance(centers[i]))for i in range(len(centers))]
    return centers[scores.argmax()]

class RegionSelector():
    def detect(self, image):
        pass
    
class PrimitiveROI(RegionSelector):
    
    def detect(self, image):
        h,w,_ = image.shape
        return np.uint8(np.zeros(shape=(h,w)))
    
    def __str__(self):
        return self.__class__.__name__

class IntervalSkinDetector(RegionSelector):
    
    def _skin_intervals(self, image):
        ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = 0
        unary = np.logical_or.reduce((ycrcb[:,:,1] >= 181, ycrcb[:,:,1] <= 137, ycrcb[:,:,2] >= 126, ycrcb[:,:,2] <= 74))
        return np.uint8(unary)
    
    def detect(self, image):
        hmap = self._skin_intervals(image)
#         hmap = cv.morphologyEx(hmap, cv.MORPH_OPEN, np.ones((8,8), np.uint8))
        return hmap
    
    def __str__(self):
        return self.__class__.__name__



class RepeatedKMeansSkinDetector(RegionSelector):
    
    def detect(self, frame):
        image = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        image[:, :, 0] = 0
        h,w,_ = image.shape
        arr = image.reshape((h*w,3))
        kmeans = KMeans(n_clusters=2, n_jobs=-1, max_iter=50).fit(arr)
        counts = np.bincount(kmeans.labels_)
        dominant = counts.argmax()
        labels = kmeans.labels_
        if dominant == 1:
            labels = np.uint8(labels==0)
        labels = labels.reshape(*(image.shape[:2]))
        return np.uint8(labels)
    
    def __str__(self):
        return self.__class__.__name__

class BayesianSkinDetector(RegionSelector):

    def __init__(self, threshold=0.5, skin_tone_freshness=30, mean=0, std=20):
        self.threshold = threshold
        self.std=std
        self.mean=mean
        self._frame_number = 0
        self._freshness = skin_tone_freshness
        self.skin_tone = None
        self.distribution = None

    def _prior(self, image):
        return 

    def _class_conditional(self, image, skin_tone):
        # DO this vectorized!!!! 
        numerator = _pdf(_dist(image, skin_tone))
        return

    def _max_distance(self, skin_tone):

    def _dist(self, x, y):
        return np.sqrt(np.sum(np.square(x-y)))

    def _pdf(self, x):
        return self.distribution.pdf(x)

    def detect(self, image):
        if(self.frame_number % self._freshness == 0):
            self.skin_tone = skin_tone(image)
            self.distribution = truncnorm(a=0, loc=0, b=_max_distance(self.skin_tone), )
        self.frame_number += 1
        return (self._class_conditional(image)*self._prior(image))>self.threshold
    
    def __str__(self):
        return f"{self.__class__.__name__}_mean-{self.mean}_std-{self.std}_threshold-{self.threshold}"