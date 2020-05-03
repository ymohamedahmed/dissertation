import cv2 as cv
import glob
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from face_det import DNNDetector
from configuration import PATH

def get_test_images():
    return [cv.imread(file, cv.IMREAD_UNCHANGED) for file in glob.glob(f"{PATH}test-roi-images/*.jpg")]

def get_eval_images():
    return [cv.imread(file, cv.IMREAD_UNCHANGED) for file in glob.glob(f"{PATH}test-roi-images/*.jpeg")]

def get_cropped_test_images():
    images = []
    for i in get_test_images():
        for (x,y,w,h) in DNNDetector().detect_face(i):
            images.append(i[y:y+h, x:x+w])
    return images 

def display_heatmap(images):
    cols = 1
    sns.set(font_scale=3)
    for n, image in enumerate(images):
        fig, ax = plt.subplots(figsize=(10,10))
        sns.set_style("white")
        ax = sns.heatmap(image, ax=ax, xticklabels=[], yticklabels=[], cbar_kws={'label': 'Probability of being a skin pixel', 'shrink':1})
    fig.tight_layout()
    h,w = images[0].shape
    impl_path = "/Users/yousuf/Workspace/dissertation/report/implementation/"
    plt.savefig(f"{impl_path}hmap-{h}-{w}.png", bbox_inches="tight", pad_inches=0)
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
