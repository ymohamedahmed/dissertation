import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy.stats import truncnorm, truncexpon
from functools import partial
from helper import display_heatmap
import pandas as pd
from configuration import PATH

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

def skin_tone(frame, alpha=1, beta=5, clusters=5):
    image = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    image[:, :, 0] = 0
    h,w,_ = image.shape
    arr = image.reshape((h*w,3))
    kmeans = KMeans(n_clusters=clusters, n_jobs=-1, max_iter=50).fit(arr)
    centers = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    scores = np.array([(alpha*counts[i]) - (beta*_cluster_skin_distance(centers[i])) for i in range(len(centers))])
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
        self.prior = self._load_prior()
    
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
        # new_data[:,0] = 0
        prior = new_data.groupby(by=["Cr", "Cb"]).agg({"Skin":[np.mean, len]})
        lookup = 0.3*np.ones(shape=(256,256))
        for (x,y) in prior.index:
            lookup[int(x),int(y)] = 1-prior.loc[x,y]["Skin"]["mean"]
        return lookup

    def _safe_prior_lookup(self, colour):
        p =  self.prior[colour[1], colour[2]]
        if p == 0: return 0.3
        return p
        # try:
        #     p = self.prior.loc[colour[1],colour[2]]
        #     return p["Skin"]["mean"]
        # except:
        #     return 0.3

    def _prior(self, image):
        return np.apply_along_axis(self._safe_prior_lookup, 2, image)
        # np.array(list(map(_safe_prior_lookup)))
        # return 0.7

    def _class_conditional(self, image, skin_tone):
        # DO this vectorized!!!! 
        # print(f"Image: {image}")
        # print(f"Skin tone: {skin_tone}")
        distance_matrix = self._dist(image, skin_tone, axis=2)
        print(f"Distance matrix: {distance_matrix}")
        display_heatmap([distance_matrix])
        self.skin_std = 0.5*np.mean(distance_matrix)
        print(f"Std: {self.skin_std}")
        skin_probs = np.array(list(map(partial(self._pdf, self.skin_std, np.max(distance_matrix)), distance_matrix)))
        # numerator = np.array(list(map(self._pdf, distance_matrix)))

        # x = np.zeros(256**2)
        # y,z = np.mgrid[0:256:1, 0:256:1]
        # all_skin_tones = np.vstack((x, y.flatten(), z.flatten())).T
        # denominator = np.sum(self._pdf(self._dist(all_skin_tones, image, axis=2)))

        # print(f"Numerator: {numerator}, Denominator: {denominator}")

        not_skin_dm = self._max_distance(skin_tone)-distance_matrix
        # display_heatmap(_cluster_skin_distance(image))

        print("Cluster skin distance")
        display_heatmap([np.apply_along_axis(_cluster_skin_distance, 2, image)])
        print("Skin probability")
        display_heatmap([skin_probs])
        # display_heatmap([skin_probs/denominator])
        print("Not skin distance matrix")
        display_heatmap([not_skin_dm])
        not_skin_probs = np.array(list(map(partial(self._pdf, 1.2*self.skin_std, np.max(not_skin_dm)), not_skin_dm)))
        # not_skin_probs = 1-skin_probs
        # not_skin_denominator = np.sum(self._pdf(self._dist(all_skin_tones, skin_tone)))
        print("Not skin probability")
        display_heatmap([not_skin_probs])
        denominator = (skin_probs+not_skin_probs)#*(1/(256**2))
        # print(f"Skin probs: {skin_probs} \n non-skin probs: {not_skin_probs} \n denominator: {denominator}")
        print("Weighted skin probability")
        display_heatmap([skin_probs/denominator])
        return denominator, not_skin_probs/denominator
        # return numerator/denominator
    
    def _max_distance(self, skin_tone):
        # Since there are eight corners of the cube, but Y is zero for all of them 
        # since we're not considering the Y value in the YCrCb colour space
        corners = np.array([[0,0,0], [0,0,255], [0,255,0], [0,255,255]])
        return np.max(self._dist(corners, skin_tone, axis=1))

    def _dist(self, x, y, axis=1):
        return np.sqrt(np.sum(np.square(x-y), axis=axis))

    def _pdf(self, std, upper_limit, x):
        # return truncnorm.pdf(x=x, a=0, loc=self.mean, b=self._max_distance(self.skin_tone), scale=std)
        return truncnorm.pdf(x=x, a=0, loc=self.mean, b=upper_limit, scale=std)
        # return truncexpon.pdf(x=x, b=upper_limit, scale=100)

    def detect(self, image):
        image = cv.GaussianBlur(image,(5,5),cv.BORDER_DEFAULT) 
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        image[:, :, 0] = 0
        if(self._frame_number % self._freshness == 0):
            self.skin_tone = skin_tone(image)
        self._frame_number += 1
        skin_probs, ns_probs = self._class_conditional(image, self.skin_tone)
        prior = self._prior(image)
        skin_post, ns_post = skin_probs*prior, ns_probs*(1-prior)
        print("Prior heatmap")
        display_heatmap([prior])
        print(skin_probs + ns_probs)
        print("Should be all ones")
        print(skin_post + ns_post)
        skin_post = skin_post*1/np.max(skin_post)
        return skin_post*1/np.max(skin_post), ns_post, skin_post>ns_post, skin_post>0.5
        # return (*self._prior(image))#>self.threshold
    
    def __str__(self):
        return f"{self.__class__.__name__}_mean-{self.mean}_skin-std-{self.skin_std}_threshold-{self.threshold}"