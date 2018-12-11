# MotionDetector

This motion detector compare 2 images in a thin time interval and detect the changes.
The code is inspired by the tutorial https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

The changes are counted by the number of white pixels and the normalized value (between 0 and 1) is send to ROS Topic by a Float64 variable.
