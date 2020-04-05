from configuration import PATH
import pandas as pd
import numpy as np
import time
import cv2 as cv
from scipy.stats import truncnorm

def compute_prior():
    print("Computing prior")
    dataset_path = f"{PATH}skin/Skin_NonSkin.txt"
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
    lookup = 0.9*np.ones(shape=(256,256))
    for (x,y) in prior.index:
        lookup[int(x),int(y)] = max(1-prior.loc[x,y]["Skin"]["mean"],0.9)
    np.savetxt(f"{PATH}/skin/prior.txt", lookup)
    print("Prior saved")

def dist(x, y):
    return np.sqrt(np.sum(np.square(x-y), axis=1))
    
def _max_distance(skin_tone):
    """
        Measure distance against each of the corners
    """
    corners = np.array([[0,0], [0,255], [255,0], [255,255]])
    return np.max(dist(corners, skin_tone))

def compute_all_truncnorm_pdfs():
    for cr in range(256):
        for cb in range(256):
            print("======================")
            print(f"Considering: ({cr},{cb}), progress = {100*((cr*256)+cb)/(256**2)}")
            skin_tone = np.array([cr,cb])
            output_file = f"{PATH}skin/{cr}-{cb}.txt"
            max_dist = _max_distance(skin_tone)
            posterior = np.zeros(shape=int(max_dist))
            rv = truncnorm(a=0, b=max_dist, scale=10)
            for i in range(int(max_dist)):
                posterior[i] = rv.pdf(x=i)
            np.savetxt(output_file, posterior)

if __name__ == "__main__":
    compute_prior()
    compute_all_truncnorm_pdfs()
    pass