import argparse
import asyncio
import datetime
import queue
import sys
import threading
import time
from queue import Queue
from threading import Thread

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
ap.add_argument("--model", type=str, default="mobilenet_thin", help="cmu/mobilenet_thin/mobilenet_v2_large/mobilenet_v2_small")
args = ap.parse_args()

# initiate camera
if args.video is None:
    camera = cv2.VideoCapture(1)
else:
    camera = cv2.VideoCapture(args["video"])
_, frame = camera.read()

# model target size
size = frame.shape[1::-1]

# simplified timer caller
now = lambda: time.time()
lineno = lambda: sys._getframe().f_back.f_lineno

class MyThread(Thread):
    def __init__(self, target, args=()):
        Thread.__init__(self)
        self.target = target
        self.args = args

    def run(self):
        self.result = self.target(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

# initiate tf_pose model
Estimator = TfPoseEstimator(get_graph_path(args.model), target_size=size)

# read frame
def read_camera(camera):
    _, frame = camera.read()
    return frame

def call_model(frame):
    print("running tf_pose, it will return the human skeleton data.")
    humans = Estimator.inference(frame)
    return humans

# rendering
def render(frame, humans):
    try:
        frame = Estimator.draw_humans(frame, humans)
    except Exception as e:
        print("line", lineno(), ", render(frame, humans): ", e)
        pass
    return frame

# show frame
def display(frame):
    cv2.imshow("camera", frame)

# main function
def main():
    maxsize = 5
    queue_task = Queue(maxsize)
    tmpbody = None
    for i in range(maxsize - 1):
        frame = read_camera(camera)
        t = MyThread(target=call_model, args=(frame,))
        queue_task.put(t)
        t.start()

    t0 = now()
    try:
        while True:
            frame = read_camera(camera)
            if now() - t0 > 0.5:
                t0 = now()
                t = MyThread(target=call_model, args=(frame,))
                t.setDaemon(True)
                t.start()
                try:
                    queue_task.put(t)
                except queue.Full:
                    pass
                try:
                    # get return value from thread
                    body = queue_task.get_nowait().get_result()
                except queue.Empty:
                    body = tmpbody
                render(frame, body)
            else:
                render(frame, body)
            display(frame)
            print(queue_task.qsize())

            # to keep showing video
            amp = 0xFF
            key = cv2.waitKey(1) & amp

    except KeyboardInterrupt:
        print("Quit...")
        while not queue_task.empty:
            queue_task.get_nowait()

        # final clear
        camera.release()
        cv2.destroyAllWindows()
        print("Quit...done")
        
if __name__ == "__main__":
    main()
    exit()
