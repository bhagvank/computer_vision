
from collections import deque
from imutils.video import VideoStream
import numpy as nump
import argparse as parser
import cv2 as opencv2
import imutils
import time


ap = parser.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
	vs = VideoStream(src=0).start()

else:
	vs = opencv2.VideoCapture(args["video"])


time.sleep(2.0)

while True:
	frame = vs.read()

	frame = frame[1] if args.get("video", False) else frame

	if frame is None:
		break


	frame = imutils.resize(frame, width=600)
	blurred = opencv2.GaussianBlur(frame, (11, 11), 0)
	hsv = opencv2.cvtColor(blurred, opencv2.COLOR_BGR2HSV)


	mask = opencv2.inRange(hsv, greenLower, greenUpper)
	mask = opencv2.erode(mask, None, iterations=2)
	mask = opencv2.dilate(mask, None, iterations=2)


	cnts = opencv2.findContours(mask.copy(), opencv2.RETR_EXTERNAL,
		opencv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None


	if len(cnts) > 0:

		c = max(cnts, key=opencv2.contourArea)
		((x, y), radius) = opencv2.minEnclosingCircle(c)
		M = opencv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


		if radius > 10:

			opencv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			opencv2.circle(frame, center, 5, (0, 0, 255), -1)


	pts.appendleft(center)


	for i in range(1, len(pts)):

		if pts[i - 1] is None or pts[i] is None:
			continue


		thickness = int(nump.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		opencv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


	opencv2.imshow("Frame", frame)
	key = opencv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break


if not args.get("video", False):
	vs.stop()


else:
	vs.release()


opencv2.destroyAllWindows()