import cv2 as cv
import glob
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from face_det import DNNDetector

def get_test_images():
    return [cv.imread(file, cv.IMREAD_UNCHANGED) for file in glob.glob("/home/yousuf/workspace/dissertation/code/rPPG/test-roi-images/*.jpg")]

def get_cropped_test_images():
    images = []
    for i in get_test_images():
        for (x,y,w,h) in DNNDetector().detect_face(i):
            images.append(i[y:y+h, x:x+w])
    return images 

def display_heatmap(images):
    cols = 1
    fig = plt.figure()
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for n, image in enumerate(images):
        a = fig.add_subplot(cols, np.ceil(len(images)/float(cols)), n + 1, aspect="equal")
#         plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        sns.set_style("white")
        ax = sns.heatmap(image, ax=a, xticklabels=[], yticklabels=[], cbar=not(n), cbar_ax = None if n else cbar_ax, cbar_kws={'label': 'Probability of being a skin pixel'})
        plt.axis("off")
    fig.set_size_inches(1.05*np.array(fig.get_size_inches()) * len(images))
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()

def show_images_plt(images):
  cols = 1
  fig = plt.figure()
  for n, image in enumerate(images):
    a = fig.add_subplot(cols, np.ceil(len(images)/float(cols)), n + 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis("off")
  fig.set_size_inches(1.1*np.array(fig.get_size_inches()) * len(images))
  plt.show()
