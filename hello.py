import numpy as np
import cv2

image = cv2.imread('Contours.jpg')
imagegray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,threshold = cv2.threshold(imagegray,127,255,0)
contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("contours",contours)