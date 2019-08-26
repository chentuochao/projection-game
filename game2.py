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

import cv2
import numpy as np
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
# if args.video == '' and args.images == '':
#     raise ValueError('Either --video or --image has to be provided')

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
    print("connecting")
    s.connect(addr) 
    print("Connect Succefully")
    def handle(sx):     # sx表示第sx个客户端，接受信息
        global ans
        while True:
            if end == 1:
                sx.close()
                break
            ans = sx.recv(1)
            print('message from client')

    t = threading.Thread(target=handle, args=(s,))   # 开启一个新的线程专门负责当前客户端数据接收, changed 'args' from () to (s,)
    return t

# Global variables
print("pygame initializing...")
pygame.init()   # pygame初始化
cap = cv2.VideoCapture(0)    # initialize the camera
size = width,height = 1920,1080 # 设置屏幕尺寸1920,1080
pic_size = 640, 480
# Define colors
BLUE   = 0,0,255
WHITE  = 255,255,255
BLACK  = 0,0,0
RED    = 255,0,0
GREEN  = 0,255,0

# screen = pygame.display.set_mode(size) # 创建surface对象
# pygame.display.set_caption('Projection game') # 创建标题

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

# 转换系数
a=[]
b=[]

def initialization(width = 1920, height = 1080):
    size=1920, 1080
    q1=(540,120)
    q2=(140,630)
    q3=(840,630)
    draw_circle(q1,q2,q3)

    Get_picture(0)   # 0 is the number of picture

    p1=get_pos()
    p2=get_pos()
    p3=get_pos()
    while calibration(q1, q2, q3, p1, p2, p3) is not True:
        p1=get_pos()
        p2=get_pos()
        p3=get_pos()
        # calibration(q1,q2,q3,p1,p2,p3)    # calculate the calbration cofficient


    # test the calibratiob
    # 左上，右上，右下，左下，中间
    test_q= ((0,0), (width,0), (width,height), (0,height), (width//2,height//2))
    errorx = 50
    errory = 50
    draw_grids(test_q)
    print("p1")
    Get_picture(1)
    print("p1")
    for num in range(0, 0):
        print(num)
        p = convert_position(get_pos())
        print(p)
        if abs(p[1]-test_q[num][1]) > errory or abs(p[0]-test_q[num][0]) > errorx:
            print("Calibration Error: Fail to acquire an accurate calibration!")
            sys.exit()
            raise KeyboardInterrupt  # 退出游戏
    screen = pygame.display.set_mode(size)
    screen.fill(WHITE) # 填充屏幕
    pygame.display.flip()
    return screen

# -----------------------coordinate converting function-----------------------
def calibration(q1, q2, q3, p1, p2, p3):
    global a
    global b

    p=np.array([ [ p1[0],p1[1],1 ],[ p2[0], p2[1],1 ],[ p3[0], p3[1],1 ] ])
    qx=np.array( [ [q1[0]], [q2[0]], [q3[0]] ])
    qy=np.array( [ [q1[1]], [q2[1]], [q3[1]] ])

    try:
        p_inv = np.linalg.inv(p)
        a = np.matmul(p_inv, qx)
        b = np.matmul(p_inv, qy)
        print(a)
        return True
    except np.linalg.LinAlgError as e:
        print(e)
        return False

def convert_position(position):    # convert the coordinate of camera to the coordinate of projection 
    global a
    global b

    # p = np.array([float(p1[0], p1[1], 1]))
    # # print('===')
    # # print(a)
    # qx = np.matmul(p, a)
    # qy = np.matmul(p, b)
    # return([qx, qy])
    """
    param: position shape: n*2
    return: points shape: n*2
    """
    points = []
    for pos in position:
        p = np.array([float(pos[0]), float(pos[1]), 1.])
        points.append([np.matmul(p, a), np.matmul(p, b)])

    return points
# ------------------------------------------------------------------------------

# The function for intialization and calibration
def draw_circle(q1,q2,q3,size=(1920,1080)):   # 画图函数
    screen = pygame.display.set_mode(size)# ,pygame.FULLSCREEN)
    screen.fill(WHITE) # 填充屏幕
    pygame.draw.circle(screen, BLACK, q1, 20, 0) 
    pygame.draw.circle(screen, BLACK, q2, 20, 0) 
    pygame.draw.circle(screen, BLACK, q3, 20, 0) 

    pygame.display.flip()

def draw_grids(q,size=(1920,1080)):   # 测试函数
    screen = pygame.display.set_mode(size)# ,pygame.FULLSCREEN)
    screen.fill(WHITE) # 填充屏幕
    divide=10
    for i in range(1,divide):   # horizon
        pygame.draw.line(screen, BLACK, (0, height*i//divide), (width,height*i//divide), 2)

    for i in range(1,divide):   # vertical
        pygame.draw.line(screen, BLACK, (width*i//divide,0), (width*i//divide, height), 2)

    pygame.draw.circle(screen, RED, q[0], 20, 0) 
    pygame.draw.circle(screen, RED, q[1], 20, 0) 
    pygame.draw.circle(screen, RED, q[2], 20, 0) 
    pygame.draw.circle(screen, RED, q[3], 20, 0) 
    pygame.draw.circle(screen, RED, q[4], 20, 0) 
    pygame.display.flip()

def get_pos():   # return the position where the mouse clicks 
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            if event.type == pygame.MOUSEBUTTONDOWN: # 获取点击鼠标事件
                if event.button == 1:  # 点击鼠标左键
                    mouse_position = pygame.mouse.get_pos()
                    # print(mouse_position)
                    return mouse_position

def Get_picture(i):     # get, save and visualize a picture from camera
    while (1):
        ret, frame = cap.read()
        flag=0
        # print(flag)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                pic_size = frame.shape[1::-1]
                # print(frame.shape)
                cv2.imwrite('./' + str(i) + '.jpg', frame)
                flag = 1
        if flag==1:
            break

    print("Get pic!!")
    # print(pic_size)
    screen = pygame.display.set_mode(pic_size)
    screen.fill(WHITE)
    space = pygame.image.load('./' + str(i) + '.jpg')
    screen.blit(space, (0, 0))
    # 3.刷新游戏窗口
    pygame.display.update()
   
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

def mydraw(screen):   # 画图函数
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

def main(net, height_size, cpu, track_ids, ans, end, screen):
    position  = [size[0] // 2 , size[1] // 2]  
    direction = -90
    color     = BLACK
    ifbound   = 1       # border of screen
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
            _, _, left_wrists, right_wrists = video.run(net, cap.read()[1], height_size, cpu, track_ids)
            hand_position = np.append(left_wrists, right_wrists, axis=0)
            # -------------------code above------------------------
            hand_position_convert = convert_position(hand_position)
            print(hand_position.shape)
            match(hand_position_convert)
            ans = b'0'

        now = time.time()
        t = now - past
        past = now
        print("running...")
        for shape in shape_tuple:   
            move(shape, t)          # movement, update shapes' position & accelarate, update shapes' velocity
        mydraw(screen)
        time.sleep(0.1)

    
if __name__ == '__main__':

    print("start __main__")

    screen = initialization()
    print("pygame initialization success!")
    # connect().start()

    # init body recognition model
    net = video.PoseEstimationWithMobileNet()
    checkpoint = video.torch.load(args.checkpoint_path, map_location='cpu')
    video.load_state(net, checkpoint)

    print("loading cv model success!")

    # start main function
    try:
        main(net, args.height_size, args.cpu, args.track_ids, ans, end, screen)
    except KeyboardInterrupt:
        cap.release()
        end = 1
        print('Goodbye')
