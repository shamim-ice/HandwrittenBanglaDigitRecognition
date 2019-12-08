import cv2
from cv2 import *
import numpy as np
from skimage.exposure import rescale_intensity


class SimplePreprossesing:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def Preprocess(self, image):
        image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        #gauss = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 15)
        #_,image=cv2.threshold(gauss,90,255,cv2.THRESH_BINARY_INV)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #cv2.imshow('Image', image)
        #cv2.waitKey(0)
        return image
