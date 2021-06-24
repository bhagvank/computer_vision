import cv2
import numpy as np

img = cv2.imread('plan.jpg')

img=cv2.resize(img,(1700,700))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=255-gray
gray=cv2.threshold(gray,4,255,cv2.THRESH_BINARY)[1]
gray=cv2.blur(gray,(15,1))
contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_NONE )

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>150000 and area<500000:
        cv2.drawContours(img,[cnt],0,(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey()
