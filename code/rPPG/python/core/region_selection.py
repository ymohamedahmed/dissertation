import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.stats import truncnorm, truncexpon
from functools import partial
from helper import display_heatmap
import pandas as pd
import time
from configuration import PATH

def _cluster_skin_distance(ycrcb_colour):
    """
        We define skin threshold as (137, 74) => (181, 126) and compute distance from a point to this rectangle
    """
    x,y = (137+181)/2.0, (74+126)/2.0
    width, height = 181-137, 126-74
    px,py = ycrcb_colour[0], ycrcb_colour[1]
    dx = max(abs(px - x) - width / 2, 0)
    dy = max(abs(py - y) - height / 2, 0)
    return (dx * dx) + (dy * dy)

def skin_tone(frame, alpha=10, beta=1, clusters=5):
    h,w,_ = frame.shape
    arr = frame.reshape((h*w,2))
    kmeans = KMeans(n_clusters=clusters, n_jobs=-1, max_iter=1000).fit(arr)
    centers = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    scores = np.array([(alpha*counts[i]) - (beta*_cluster_skin_distance(centers[i])) for i in range(len(centers))])
    return centers[scores.argmax()]

def mean(frame, distribution):
    distribution = distribution/np.sum(distribution)
    return np.array([np.sum(distribution*frame[:,:,i]) for i in range(3)])

class RegionSelector():
    def detect(self, image):
        pass
    
class PrimitiveROI(RegionSelector):
    """
        The simplest region selector which considers the entire bounding box around the face
    """
    
    def detect(self, image):
        h,w,_ = image.shape
        mask = np.uint8(np.ones(shape=(h,w)))
        return mask, mean(image, mask)
    
    def __str__(self):
        return self.__class__.__name__

class IntervalSkinDetector(RegionSelector):
    """
        Rudimentary skin detector, assumes any pixel in the known range of skin pixels is skin.
    """
    
    def _skin_intervals(self, image):
        ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = 0
        unary = np.logical_and.reduce((ycrcb[:,:,1] <= 181, ycrcb[:,:,1] >= 137, ycrcb[:,:,2] <= 126, ycrcb[:,:,2] >= 74))
        return np.uint8(unary)
    
    def detect(self, image):
        hmap = self._skin_intervals(image)
        return hmap, mean(image,hmap)
    
    def __str__(self):
        return self.__class__.__name__



class RepeatedKMeansSkinDetector(RegionSelector):
    """
        Applies k-means to every frame and assumes the largest cluster corresponds to skin
    """
    
    def detect(self, frame, clusters=2):
        image = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        image[:, :, 0] = 0
        h,w,_ = image.shape
        arr = image.reshape((h*w,3))
        kmeans = KMeans(n_clusters=clusters, n_jobs=-1, max_iter=50).fit(arr)
        counts = np.bincount(kmeans.labels_)
        dominant = counts.argmax()
        labels = np.uint8(kmeans.labels_==dominant)
        labels = labels.reshape(*(image.shape[:2]))
        return np.uint8(labels), mean(frame, labels)
    
    def __str__(self):
        return self.__class__.__name__

class BayesianSkinDetector(RegionSelector):
    """
        Conditionalise the colour of the pixels on the skin tone of the user
    """

    def __init__(self, mean=0, skin_std=20, non_skin_std=100, weighted=True):
        self.skin_std=skin_std
        self.non_skin_std=non_skin_std
        self.mean=mean
        self._frame_number = 0
        self.skin_tone = None
        self.distribution = None
        self.prior_lookup = self._load_prior()
        self.weighted = weighted
        self.prior = None
        self.class_conditional = None
    
    def _load_prior(self):
        return np.loadtxt(f"{PATH}skin/prior.txt")
    
    def _load_class_conditional(self, skin_tone):
        cr, cb = int(skin_tone[0]), int(skin_tone[1])
        return np.loadtxt(f"{PATH}skin/{cr}-{cb}.txt")

    def _prior(self, image):
        return self.prior_lookup[image[:,:,0], image[:,:,1]]

    def _class_conditional_lookup(self, image, skin_tone):
        start = time.time()
        distance_matrix = self._dist(image, skin_tone)
        return self.class_conditional[np.uint8(distance_matrix)]

    def _dist(self, x, y, axis=2):
        return np.sqrt(np.sum(np.square(x-y), axis=axis))
    
    def _update_prior(self, image, posterior):
        self.prior_lookup[image[:,:,0], image[:,:,1]] = posterior

    def detect(self, frame):
        try: 
            image = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
            image = cv.GaussianBlur(image,(5,5),cv.BORDER_DEFAULT) 
        except Exception as e: 
            print("CONVERSION COLOUR")
            print(e)
            print(frame.shape)

        image = image[:, :, 1:3] 
        if(self._frame_number == 0):
            start = time.time()
            self.skin_tone = skin_tone(image)
            self.class_conditional = self._load_class_conditional(self.skin_tone)
            print(f"Frame: {self._frame_number}, time to find skin tone: {time.time()-start}")
        self._frame_number += 1


        start = time.time()
        skin_probs = self._class_conditional_lookup(image, self.skin_tone)

        start = time.time()
        self.prior = self._prior(image)

        skin_post = skin_probs*self.prior
        skin_post = np.minimum(skin_post*(1/np.mean(skin_post)), np.ones(skin_post.shape))
        threshold = np.percentile(skin_post, 0.2)
        start = time.time()
        self._update_prior(image, skin_post)
        mask = skin_post > threshold
        return skin_post, mean(frame, skin_post if self.weighted else mask)
    
    def __str__(self):
        return f"{self.__class__.__name__}-{'weighted' if self.weighted else 'thresholded'}"