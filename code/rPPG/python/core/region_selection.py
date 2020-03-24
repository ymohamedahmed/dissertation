import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.stats import truncnorm
from functools import partial

def _cluster_skin_distance(ycrcb_colour):
    """
        We define skin threshold as (137, 74) => (181, 126) and compute distance from a point to this rectangle
    """
    x,y = (137+181)/2.0, (74+126)/2.0
    width, height = 181-137, 126-74
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
    scores = np.array([(alpha*counts[i]) - (beta*_cluster_skin_distance(centers[i]))for i in range(len(centers))])
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

    def __init__(self, threshold=0.5, skin_tone_freshness=30, mean=0, skin_std=20, non_skin_std=100):
        self.threshold = threshold
        self.skin_std=skin_std
        self.non_skin_std=non_skin_std
        self.mean=mean
        self._frame_number = 0
        self._freshness = skin_tone_freshness
        self.skin_tone = None
        self.distribution = None

    def _prior(self, image):
        return 0.7

    def _class_conditional(self, image, skin_tone):
        # DO this vectorized!!!! 
        # print(f"Image: {image}")
        # print(f"Skin tone: {skin_tone}")
        distance_matrix = self._dist(image, skin_tone, axis=2)
        # print(f"Distance matrix: {distance_matrix}")
        skin_probs = np.array(list(map(partial(self._pdf, self.skin_std), distance_matrix)))
        # numerator = np.array(list(map(self._pdf, distance_matrix)))

        # x = np.zeros(256**2)
        # y,z = np.mgrid[0:256:1, 0:256:1]
        # all_skin_tones = np.vstack((x, y.flatten(), z.flatten())).T
        # denominator = np.sum(self._pdf(self._dist(all_skin_tones, image, axis=2)))

        # print(f"Numerator: {numerator}, Denominator: {denominator}")


        not_skin_dm = self._max_distance(skin_tone)-distance_matrix
        not_skin_probs = np.array(list(map(partial(self._pdf, self.non_skin_std), not_skin_dm)))
        # not_skin_denominator = np.sum(self._pdf(self._dist(all_skin_tones, skin_tone)))

        return skin_probs, not_skin_probs
        # return numerator/denominator
    
    def _max_distance(self, skin_tone):
        # Since there are eight corners of the cube, but Y is zero for all of them 
        # since we're not considering the Y value in the YCrCb colour space
        corners = np.array([[0,0,0], [0,0,255], [0,255,0], [0,255,255]])
        return np.max(self._dist(corners, skin_tone, axis=1))

    def _dist(self, x, y, axis=1):
        return np.sqrt(np.sum(np.square(x-y), axis=axis))

    def _pdf(self, std, x):
        return truncnorm.pdf(x=x, a=0, loc=self.mean, b=self._max_distance(self.skin_tone), scale=std)

    def detect(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        image[:, :, 0] = 0
        if(self._frame_number % self._freshness == 0):
            self.skin_tone = skin_tone(image)
        self._frame_number += 1
        skin_probs, ns_probs = self._class_conditional(image, self.skin_tone)
        skin_post, ns_post = skin_probs*self._prior(image), ns_probs*(1-self._prior(image))
        return skin_post, ns_post, skin_post>ns_post
        # return (*self._prior(image))#>self.threshold
    
    def __str__(self):
        return f"{self.__class__.__name__}_mean-{self.mean}_skin-std-{self.skin_std}_threshold-{self.threshold}"