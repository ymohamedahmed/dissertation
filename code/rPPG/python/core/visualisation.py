import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.cm as cm

class Visualiser():

  def __init__(self, path, tracker, video_path, roi, signal_processor, video, crop=False, file_path=None, width=None, height=None):
    if file_path is None: 
      input_video_name = video_path.split("/")[-1].split(".")[-2]
      self.file_path = f'{path}output/{input_video_name}-{str(tracker)}-{str(roi)}-{str(signal_processor)}.avi'
    else: 
      self.file_path = file_path
    if crop is None and None in [width,height]:
      raise ValueError("If crop is desired, then width and height must be specified")
    self.frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    self.frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    self.tracker = tracker
    self.crop = crop
    self.out = cv.VideoWriter(self.file_path,cv.VideoWriter_fourcc('M','J','P','G'), video.get(cv.CAP_PROP_FPS), (self.frame_width,self.frame_height))
    self.cmap = cm.get_cmap('RdBu')


  def display(self, frame, faces, area_of_interest):
      height, width, _ = frame.shape
      self.tracker.overlay(frame, faces)
      x,y,w,h = faces[0]

      area_of_interest = np.pad(np.ones(shape=(h,w)), ((y,height-(y+h)),(x,width-(x+w))), 'constant', constant_values=0)
      rectangle = np.array(list(map(self.cmap, area_of_interest)))
      area_of_interest = np.repeat(area_of_interest[:, :, np.newaxis], 3, axis=2)

      # Overlay the points being considered and the rectangle of the face
      # rectangle = np.full(shape=(height,width,3), fill_value=[0,255,0], dtype=np.uint8)
      rectangle = rectangle[:,:,:3]
      # Need to take and with one vector since otherwise we get 254 instead of the desired value of 0 at each point outside the mask
      # i.e. NOT(0000 0001) = 1111 1110 rather than 0
      mask_of_roi = area_of_interest
      # mask_of_roi = cv.bitwise_and(cv.bitwise_not(area_of_interest), np.ones(shape=(height,width,3), dtype=np.uint8))
      alpha = 0.5
      foreground = rectangle.astype(float)
      background = frame.astype(float)
      mask_of_roi = mask_of_roi.astype(float)/2
      # print(mask_of_roi.shape)
      # print(rectangle.shape)
      foreground = cv.multiply(mask_of_roi, foreground)
      background = cv.multiply(1.0-mask_of_roi, background)
      blended = cv.addWeighted(foreground, alpha, background, 1, 0)
      if self.crop:
        blended = cv.resize(blended, (self.frame_width, self.frame_height), interpolation = cv.INTER_AREA)
      return rectangle, foreground, background, blended
      # self.out.write(np.uint8(blended))
