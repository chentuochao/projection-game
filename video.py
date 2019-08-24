import argparse
import asyncio
import datetime
import time
import queue
import threading
from queue import Queue
from threading import Thread

import cv2
import imutils
import numpy as np

import baidu

# create arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# initiate camera
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
else:
    camera = cv2.VideoCapture(args["video"])

now = lambda: time.time()
client = baidu.client

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


# read frame
def read_camera(camera):
    _, frame = camera.read()
    return frame

# call baidu sdk
def call_baidu(frame):
    ret, data = cv2.imencode('.png', frame)     # correct!
    # body = baidu.client.bodyAnalysis(data)
    body = client.gesture(data)
    return body

# render in the frame
def render_body(frame, body):
    try:
        for i in body['person_info']:
            if i['location']['score'] > 0.03:
                for part in i['body_parts']:
                    x = int(i['body_parts'][part]['x'])
                    y = int(i['body_parts'][part]['y'])
                    if i['body_parts'][part]['score'] > 0.3:
                        cv2.circle(frame,(x, y), 3, (0, 145, 240), -1)
                        cv2.putText(frame, part, (x+1, y+1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (156, 144, 238))
    except Exception as e:
        print("baidu API has not returned yet!")
        print("in render_body(frame, body): ", e)
        print(body)
    return frame

def render_gesture(frame, gesture):
    try:
        for i in gesture['result']:
            if i['probability'] > 0.3 and i['classname'] != 'Face':
                top = int(i['top'])
                height = int(i['height'])
                left = int(i['left'])
                width = int(i['width'])
                cv2.rectangle(frame, (left, top), (left+width, top+height), (250, 180, 56), thickness=10)
    except Exception as e:
        print("baidu API has not returned yet!")
        print("in render_gesture(frame, gesture): ", e)
        print(gesture)
    return frame

# show frame
def display(frame):
    cv2.imshow("camera", frame)

# main function
def main():
    maxsize = 5
    queue_frame = Queue(maxsize)
    queue_task = Queue(maxsize)
    tmpbody = None
    for i in range(maxsize - 1):
        frame = read_camera(camera)
        queue_frame.put(frame)
        t = MyThread(target=call_baidu, args=(frame,))
        queue_task.put(t)
        t.start()
        time.sleep(0.5)

    t0 = now()
    try:
        while True:
            frame = read_camera(camera)
            if now() - t0 > 0.5:
                t0 = now()
                t = MyThread(target=call_baidu, args=(frame,))
                t.setDaemon(True)
                t.start()
                try:
                    queue_task.put(t)
                except queue.Full:
                    queue_task.get_nowait()
                try:
                    body = queue_task.get_nowait().get_result()
                    try:
                        tmpbody = body.copy()
                    except:
                        pass
                except queue.Empty:
                    body = tmpbody
                # render_body(frame, body)
                render_gesture(frame, body)
            else:
                # render_body(frame, tmpbody)
                render_gesture(frame, tmpbody)
            display(frame)
            print(queue_task.qsize())
            # to keep showing video
            amp = 0xFF
            key = cv2.waitKey(1) & amp

    except KeyboardInterrupt:
        print("Quit...")
        queue_task.join()
        # final clear
        camera.release()
        cv2.destroyAllWindows()
        print("Quit...done")
        
if __name__ == "__main__":
    main()