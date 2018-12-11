## License: BSD 3.


import pyrealsense2 as rs
from imutils.video import VideoStream
import argparse
import datetime
import numpy as np
import cv2
import dlib
import time
import imutils
from numpy import linalg as la
import rospy
import sys
import csv
import math

from std_msgs.msg import Float64

max_value = 1

def scale_input(value):
    global max_value
    if value > max_value:
        max_value = value
    return value/max_value

rospy.init_node('gummi', anonymous=True)
r = rospy.Rate(30)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	#vs = VideoStream(src=0).start()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    OUTPUT_SIZE_WIDTH = 775
    OUTPUT_SIZE_HEIGHT = 600
    rectangleColor = (0,165,255)
    # Start streaming
    pipeline.start(config)
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
motion_detector = rospy.Publisher("/motion_detector", Float64, queue_size=10)

try:
    while True :

        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        baseImage = cv2.resize( color_image, (640, 480))
        gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        n_white_pix = np.sum(thresh == 255)
        firstFrame = gray
        val = scale_input(n_white_pix)
        print(val)
        value_motion = Float64(val)
        motion_detector.publish(value_motion)

        #print('Number of white pixels:', n_white_pix)

        # Show images
        # Stack both images horizontally
        #images = np.hstack((color_image, largeResult))
        #cv2.namedWindow('BaseImage', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('FrameDelta', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('BaseImage', baseImage)
        cv2.imshow('Threshold', thresh)
        #cv2.imshow('FrameDelta', frameDelta)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
