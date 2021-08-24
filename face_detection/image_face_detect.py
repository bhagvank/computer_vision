import numpy as nump
import argparse
import cv2 as opencv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to inumput image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("loading the model...")
net = opencv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = opencv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = opencv2.dnn.blobFromImage(opencv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

print("computing the object detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]

	if confidence > args["confidence"]:
		box = detections[0, 0, i, 3:7] * nump.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		opencv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		opencv2.putText(image, text, (startX, y),
			opencv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

opencv2.imshow("Output", image)
opencv2.waitKey(0)