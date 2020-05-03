import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.cm as cm
import helper
import sys

class Visualiser():

  def __init__(self, path, tracker, video_path, roi, signal_processor, video, file_path=None, webcam=False):
    if not(webcam):
      if file_path is None: 
        input_video_name = video_path.split("/")[-1].split(".")[-2]
        self.file_path = f'{path}output/{input_video_name}-{str(tracker)}-{str(roi)}-{str(signal_processor)}.avi'
      else: 
        self.file_path = file_path
    self.frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    self.frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    self.tracker = tracker
    if not(webcam):
      self.out = cv.VideoWriter(self.file_path,cv.VideoWriter_fourcc('M','J','P','G'), video.get(cv.CAP_PROP_FPS), (self.frame_width,self.frame_height))
    self.cmap = cm.get_cmap('RdBu')
    self.webcam = webcam


  def display(self, frame, faces, area_of_interest, hr=None):
    height, width, _ = frame.shape
    x,y,w,h = faces[0]
    cmap = cm.get_cmap('jet')

    area_of_interest = np.pad(area_of_interest, ((y,height-(y+h)),(x,width-(x+w))), 'constant', constant_values=0)
    mask_of_roi = np.pad(np.ones(shape=(h,w)), ((y,height-(y+h)),(x,width-(x+w))), 'constant', constant_values=0)
    h,w = area_of_interest.shape
    rectangle = 255*np.array(list(map(cmap, area_of_interest)))
    rectangle = rectangle.reshape((h,w,4))
    mask_of_roi = np.repeat(mask_of_roi[:, :, np.newaxis], 3, axis=2)

    rectangle = rectangle[:,:,:3]
    alpha = 0.8
    foreground = rectangle.astype(float)
    background = frame.astype(float)
    mask_of_roi = mask_of_roi.astype(float)/2
    foreground = cv.multiply(mask_of_roi, foreground)
    background = cv.multiply(1.0-mask_of_roi, background)
    blended = cv.addWeighted(foreground, alpha, background, 1, 0)


    if hr is not None:
      cv.putText(blended,f'Heart rate: {round(hr, 4)}', 
          (10,500), 
          cv.FONT_HERSHEY_SIMPLEX, 
          1,
          (255,255,255),
          2)

    if self.webcam:
      cv.imshow('frame', np.uint8(blended))
      if (cv.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
          sys.exit()
    else:
      self.out.write(np.uint8(blended))
