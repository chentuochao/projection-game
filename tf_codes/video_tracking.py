import argparse
import datetime
import time
import sys

import cv2
import imutils
import numpy as np

import tf_pose
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# create arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("--model", type=str, default="mobilenet_thin", help="sadghfne;lf&^%^$#$%Y")
ap.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')
args = ap.parse_args()

# initiate camera
if args.video is None:
    camera = cv2.VideoCapture(1)
    time.sleep(0.25)
else:
    camera = cv2.VideoCapture(args["video"])
_, firstFrame = camera.read()
target_size = firstFrame.shape[1::-1]

now = lambda: time.time()
lineno = lambda: sys._getframe().f_back.f_lineno

e = TfPoseEstimator(get_graph_path(args.model), target_size=target_size)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break
    print(frame.shape)
    t1 = now()

    ret, data = cv2.imencode('.png', frame)     # correct!
    body = e.inference(frame)
    frame = TfPoseEstimator.draw_humans(frame, body)

    t2 = now()
    print("elapsed time: ", t2 - t1, "s")

    cv2.imshow("frame", frame)

    # to keep showing video
    amp = 0xFF
    key = cv2.waitKey(1) & amp
# final clear
camera.release()
cv2.destroyAllWindows()
