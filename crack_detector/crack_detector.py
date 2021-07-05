import argparse
import time
from PIL import Image,ImageFilter
import numpy as nump
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2 as opencv
from skimage import exposure
import urllib.request
import sys,traceback


def crack_detect(img,outputPath):
    gray = opencv.cvtColor(nump.array(img), opencv.COLOR_BGR2GRAY)
    blur = opencv.blur(gray,(3,3))
    img_log = (nump.log(blur+1)/(nump.log(1+nump.max(blur))))*255
    img_log = nump.array(img_log,dtype=nump.uint8)

    bilateral = opencv.bilateralFilter(img_log, 5, 75, 75)

    edges = opencv.Canny(bilateral,100,200)
    kernel = nump.ones((5,5),nump.uint8)
    closing = opencv.morphologyEx(edges, opencv.MORPH_CLOSE, kernel)

    orb = opencv.ORB_create(nfeatures=1500)


    keypoints, descriptors = orb.detectAndCompute(closing, None)
    featuredImg = opencv.drawKeypoints(closing, keypoints, None)
    opencv.imwrite(outputPath+'/CrackDetected.jpg', featuredImg)

if __name__ == "__main__":
    
    
    print("starting")

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    imgsize = 256
    nbands = 3

    print("starting")

    starttime = time.time()  
    print(starttime)

    parser = argparse.ArgumentParser()
    parser.add_argument('inumputPath', help='inumputPath')
    parser.add_argument('outputPath', help='outputPath')
    args = parser.parse_args()
    inumputPath = args.inumputPath
    outputPath = args.outputPath

    print(inumputPath)
    print(outputPath)


    img = Image.open(inumputPath)
    crack_detect(img,outputPath)
    
    img.filter(filter=ImageFilter.CONTOUR)
    img.save('output/Contour.jpg')


    endtime = time.time()
    print('time elapsed = ', (endtime - starttime), ' seconds.')
