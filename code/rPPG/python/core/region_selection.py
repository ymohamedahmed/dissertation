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

def skin_tone(frame, alpha=10, beta=3, clusters=5):
    h,w,_ = frame.shape
    arr = frame.reshape((h*w,2))
    kmeans = KMeans(n_clusters=clusters, n_jobs=-1, max_iter=50).fit(arr)
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
    
    def detect(self, image):
        h,w,_ = image.shape
        mask = np.uint8(np.ones(shape=(h,w)))
        return mask, mean(image, mask)
    
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
        return hmap, mean(image,hmap)
    
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
        labels = np.uint8(kmeans.labels_==dominant)
        labels = labels.reshape(*(image.shape[:2]))
        return np.uint8(labels), mean(frame, labels)
    
    def __str__(self):
        return self.__class__.__name__

class BayesianSkinDetector(RegionSelector):

    def __init__(self, threshold=0.5, mean=0, skin_std=20, non_skin_std=100, weighted=True):
        self.threshold = threshold
        self.skin_std=skin_std
        self.non_skin_std=non_skin_std
        self.mean=mean
        self._frame_number = 0
        self.skin_tone = None
        self.distribution = None
        self.prior_lookup = self._load_prior()
        self.weighted = weighted
        self.prior = None
    
    def _load_prior(self):
        dataset_path = f"{PATH}/skin/Skin_NonSkin.txt"
        data = pd.read_csv(dataset_path, sep="\t", header=None)
        data.columns = ["B", "G", "R", "Skin"]
        image = data[["B", "G", "R"]].values
        size,_ = image.shape
        image = np.uint8(image.reshape(size,1,3))
        new_data = np.zeros(shape=(size,4))
        new_data[:,:3] = cv.cvtColor(image, cv.COLOR_BGR2YCrCb).reshape(size, 3)
        new_data[:,3] = data["Skin"]-1
        new_data = pd.DataFrame(new_data, columns=["Y", "Cr", "Cb", "Skin"])
        prior = new_data.groupby(by=["Cr", "Cb"]).agg({"Skin":[np.mean, len]})
        lookup = 0.3*np.ones(shape=(256,256))
        for (x,y) in prior.index:
            lookup[int(x),int(y)] = max(1-prior.loc[x,y]["Skin"]["mean"],0.3)
        return lookup

    def _prior(self, image):
        return self.prior_lookup[image[:,:,0], image[:,:,1]]

    def _class_conditional(self, image, skin_tone):
        distance_matrix = self._dist(image, skin_tone, axis=2)
        self.skin_std = 0.5*np.mean(distance_matrix)
        skin_probs = truncnorm.pdf(x=distance_matrix, a=0, loc=self.mean, b=np.max(distance_matrix), scale=self.skin_std)
        return skin_probs
    
    def _max_distance(self, skin_tone):
        """
            Measure distance against each of the corners
        """
        corners = np.array([[0,0], [0,255], [255,0], [255,255]])
        return np.max(self._dist(corners, skin_tone, axis=1))

    def _dist(self, x, y, axis=1):
        return np.sqrt(np.sum(np.square(x-y), axis=axis))
    
    def _update_lookup(self, image, posterior):
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
            print(f"Frame: {self._frame_number}, time to find skin tone: {time.time()-start}")
        self._frame_number += 1
        skin_probs = self._class_conditional(image, self.skin_tone)
        self.prior = self._prior(image)
        skin_post = skin_probs*self.prior
        skin_post = skin_post*1/np.max(skin_post)
        threshold = np.percentile(skin_post, 0.2)
        # self._update_lookup(image, skin_post)
        mask = skin_post > threshold
        return skin_post, mean(frame, skin_post if self.weighted else mask)
    
    def __str__(self):
        return f"{self.__class__.__name__}_mean-{self.mean}_skin-std-{self.skin_std}_threshold-{self.threshold}"