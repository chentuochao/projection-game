#!/usr/bin/env python
#-*- coding:utf-8 -*- 
# Author: ss

import argparse
import os
import random
import socket
import sys
import threading
import time
from math import *

import pygame

import video
from Myshape import Myshape

# parsing commandline arguments
parser = argparse.ArgumentParser(
    description='''Lightweight human pose estimation python demo.
                    This is just for quick results preview.
                    Please, consider c++ demo for the best performance.''')
# parser.add_argument('--checkpoint-path', type=str, help='path to the checkpoint', required=True)      # distributed version
parser.add_argument('--checkpoint-path', type=str, help='path to the checkpoint', default='/media/bob-lytton/MyData/repos/torch_pose/checkpoint_iter_370000.pth')
parser.add_argument('--height-size', type=int, default=128, help='network input layer height size')     # 128 is faster than 256, and 2^n is faster with higher accuracy then others
parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
parser.add_argument('--track-ids', default=True, help='track poses ids')
args = parser.parse_args()  # get a global variable 'args', used in __main__
if args.video == '' and args.images == '':
    raise ValueError('Either --video or --image has to be provided')

ans = b'0'
end = 0
def connect():
    """
    connect to ESP8266 terminal device

    return: new thread t
    """
    addr_info = socket.getaddrinfo("192.168.4.1",80)
    addr = addr_info[0][-1]
    s = socket.socket()
    s.connect(addr) 
    def handle(sx):     # sx表示第sx个客户端，接受信息
        while True:
            if end == 1:
                break
            # ans = s.recv(1)   # what does the arg 'sx' do?
            ans = sx.recv(1)    # i guess it's used like this
            print('message from client')

    t = threading.Thread(target=handle, args=(s,))   # 开启一个新的线程专门负责当前客户端数据接收, changed 'args' from () to (s,)
    return t

# Global variables
print("pygame initializing...")
pygame.init()   # pygame初始化
size = width,height = 600,400 # 设置屏幕尺寸1920,1080
# Define colors
BLUE   = 0,0,255
WHITE  = 255,255,255
BLACK  = 0,0,0
RED    = 255,0,0
GREEN  = 0,255,0
screen = pygame.display.set_mode(size) # 创建surface对象
pygame.display.set_caption('Projection game') # 创建标题
print("pygame initialization done.")

print("sys.platform is", sys.platform)
# Define path to raw file
if sys.platform.lower().find('win') != -1:
    rawPath = '.\\raw'
elif sys.platform.lower().find('linux') != -1 or sys.platform.lower().find('darwin'):
    rawPath = './raw'
files=['luo.mp3','drum.mp3','drum2.mp3','drum3.mp3','drum4.mp3'] 
oggfilelist = []
for file in files:
    file_path = os.path.join(rawPath, file)
    oggfilelist.append(file_path)
print(oggfilelist)
pygame.mixer.init(buffer=4096) # 只初始化声音
shape_tuple=[]

cal_x  = 0
cal_y  = 0
cal_rx = 0
cal_ry = 0

def calibration():     # 初始化校准
    screen.fill(WHITE) # 填充屏幕
    pygame.draw.circle(screen, RED,  [size[0] // 2 , size[1] // 2] , 40, 0) 
    # -------------------code below------------------------
    # 返回当前视频中的圆的位置与大小,赋值给cal_x,cal_y,cal_rx,cal_ry
    # -------------------code above------------------------

def convert_position(hand_position):   # 将摄像机中的坐标转化成投影上的坐标
    """
    hand_position has left_wrists and right_wrists
    left_wrists = hand_position[0], right_wrists = hand_position[1]
    """
    posxs = []
    posys = []
    for wrists in hand_position:
        for pos in wrists:        
            posxs.append()
            posys.append()

    return (posxs, posys)

def move(shape, t):     # 投影中的形状早不断移动，物理引擎，支持加速度和速度，反弹
    boundx=[shape.radius, width-shape.radius]
    boundy=[shape.radius, height-shape.radius]
    angel=radians(shape.direction)
    acc_angel=radians(shape.A_dir)
    direction2=shape.direction
    pos2=shape.position
    pos2[0]=shape.position[0]+shape.velocity*t*cos(angel)
    pos2[1]=shape.position[1]+shape.velocity*t*sin(angel)
    if shape.ifbound == 1:
        if pos2[0]<boundx[0]:
            pos2[0] = boundx[0]*2-pos2[0]
            direction2 = 180-direction2+40*(random.random()-0.5)
        elif pos2[0]>boundx[1]:
            pos2[0] = boundx[1]*2-pos2[0]
            direction2 = 180-direction2+40*(random.random()-0.5)
        if pos2[1]<boundy[0]:
            pos2[1] = boundy[0]*2-pos2[1]
            direction2 = -1*direction2+40*(random.random()-0.5)
        elif pos2[1]>boundy[1]:
            pos2[1] = boundy[1]*2-pos2[1]
            direction2 = -1*direction2+40*(random.random()-0.5)
        shape.position=pos2
        if direction2 > 180: 
            direction2 = direction2 -360
        elif direction2 <= -180:
            direction2 = direction2 + 360
    else: 
        if pos2[0]<boundx[0] or pos2[0]>boundx[1] or pos2[1]>boundy[1] or pos2[1]<boundy[0]:
            shape_tuple.remove(shape)
            return
    angel2=radians(direction2)
    vx = shape.velocity*cos(angel2)
    vy = shape.velocity*sin(angel2)
    # print(vx,vy,direction2)
    ax = shape.acc*cos(acc_angel)
    ay = shape.acc*sin(acc_angel)
    vx = vx + ax*t
    vy = vy + ay*t
    shape.velocity = sqrt(pow(vx,2)+pow(vy,2))
    if vx==0:
        angel2 = 90
    else: angel2 = degrees(atan(vy/vx))
    if vx<0 and vy>=0:
        shape.direction=angel2+180
    elif vx<0 and vy<0:
        shape.direction=angel2-180
    else:
        shape.direction=angel2
    
    # print(vx,vy,shape.direction,direction2)
    # print(shape.direction)

def mydraw():   # 画图函数
    screen.fill(WHITE) # 填充屏幕
    # 画各种尺寸颜色的圆
    for shape in shape_tuple:
        if shape.SHAPE=='c':
            pygame.draw.circle(screen, shape.color, (int(shape.position[0]), int(shape.position[1])), shape.radius, shape.width) 
        else:
            pygame.draw.rect(screen, shape.color, (int(shape.position[0]), int(shape.position[1]), shape.radius, shape.radius), shape.width) 
    pygame.display.flip()

def match(mouse_position):   # 检测是否手在圆圈内，在的话，发出击鼓的声音
    for shape in shape_tuple:
        if shape.SHAPE=='c':
            if pow(mouse_position[0]-shape.position[0],2)+pow(mouse_position[1]-shape.position[1],2)<pow(shape.radius,2):
                print("kick")
                track1 = pygame.mixer.music.load(oggfilelist[0])
                pygame.mixer.music.stop()
                pygame.mixer.music.play()   
        elif shape.SHAPE=='r':
            if fabs(mouse_position[1]-shape.position[1])<shape.radius/2 or fabs(mouse_position[0]-shape.position[0])<shape.radius/2:
                print("kick")
                track1 = pygame.mixer.music.load(oggfilelist[1])
                pygame.mixer.music.stop()
                pygame.mixer.music.play()       

def main(net, frame_provider, height_size, cpu, track_ids, ans, end):
    position  = [size[0] // 2 , size[1] // 2]  
    direction = -90
    color     = BLACK
    ifbound   = 0
    velocity  = 200
    radius    = 40
    acc       = 0
    A_dir     = 0
    width     = 0
    shape     = Myshape(position, velocity, direction, acc, A_dir, 'c', radius, color, width, ifbound)
    shape_tuple.append(shape)

    past = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            # if event.type == pygame.MOUSEBUTTONDOWN: # 获取点击鼠标事件
                # if event.button == 1:  # 点击鼠标左键
                    # mouse_position = pygame.mouse.get_pos()
                    # match(mouse_position)
        if ans == b'1':
            # -------------------code below------------------------
            # TODO: 返回当前视频中的手的位置,赋值给hand_position
            _, _, left_wrists, right_wrists = video.run(net, frame_provider, height_size, cpu, track_ids)
            hand_position = (left_wrists, right_wrists)
            # -------------------code above------------------------
            hand_position_convert = convert_position(hand_position)
            match(hand_position_convert)
            ans = b'0'

        now = time.time()
        t = now - past
        past = now
        for shape in shape_tuple:   
            move(shape, t)          # movement, update shapes' position & accelarate, update shapes' velocity
        mydraw()
        time.sleep(0.1)

    
if __name__ == '__main__':

    print("start __main__")

    # init body recognition model
    net = video.PoseEstimationWithMobileNet()
    checkpoint = video.torch.load(args.checkpoint_path, map_location='cpu')
    video.load_state(net, checkpoint)

    frame_provider = video.ImageReader(args.images)
    if args.video != '':
        frame_provider = video.VideoReader(args.video)
    print("step 1 success!")

    # start main function
    try:
        # t, ans, end = connect()     # if test without ESP8266, comment this line and one more line below
        # t.start()
        main(net, frame_provider, args.height_size, args.cpu, args.track_ids, ans, end)
    except KeyboardInterrupt:
        end = 1
        print('Goodbye')
