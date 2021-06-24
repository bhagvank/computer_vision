import cv2
import numpy as np

img = cv2.imread('plan.jpg')
#print("image",img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=255-gray

contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_NONE )

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>9000 and area<40000:
        cv2.drawContours(img,[cnt],0,(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey()