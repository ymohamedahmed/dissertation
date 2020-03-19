import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

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

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def _prior(self, image):
        return 

    def _class_conditional(self, image, skin_tone):
        return

    def _truncated_normal_pdf(self, x, mean, std):
        return

    def detect(self, image):
        return (self._class_conditional(image)*self._prior(image))>self.threshold
    
    def __str__(self):
        return self.__class__.__name__