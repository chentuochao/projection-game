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
import numpy as np

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
    print("Connect Succefully")
    def handle(sx):     # sx表示第sx个客户端，接受信息
        global ans
        while True:
            if end == 1:
                sx.close()
                break
            # ans = s.recv(1)   # what does the arg 'sx' do?
            ans = sx.recv(1)    # i guess it's used like this
            print('message from client')

    t = threading.Thread(target=handle, args=(s,))   # 开启一个新的线程专门负责当前客户端数据接收, changed 'args' from () to (s,)
    return t

# Global variables
print("pygame initializing...")
pygame.init()   # pygame初始化
cap = cv2.VideoCapture(0)    #initialize the camera
size = width,height = 1280,800 # 设置屏幕尺寸1920,1080
pic_size = 640, 480
# Define colors
BLUE   = 0,0,255
WHITE  = 255,255,255
BLACK  = 0,0,0
RED    = 255,0,0
GREEN  = 0,255,0
# 转换系数
a=[]
b=[]
#screen = pygame.display.set_mode(size) # 创建surface对象
#pygame.display.set_caption('Projection game') # 创建标题
#-----------------------coordinate converting function-----------------------
def calibration(q1, q2, q3, p1, p2, p3):
    global a
    global b

    p=np.array([ [ p1[0],p1[1],1 ],[ p2[0], p2[1],1 ],[ p3[0], p3[1],1 ] ])
    qx=np.array( [ [q1[0]], [q2[0]], [q3[0]] ])
    qy=np.array( [ [q1[1]], [q2[1]], [q3[1]] ])

    a = np.matmul( np.linalg.inv(p), qx)
    b = np.matmul( np.linalg.inv(p), qy)

def convert_position(p1):    # convert the coordinate of camera to the coordinate of projection 
    global a
    global b

    p = np.array([p1[0], p1[1], 1])
    #print('===')
    #print(a)
    qx = np.matmul(p, a)
    qy = np.matmul(p, b)
    return([qx, qy])
#------------------------------------------------------------------------------


#The function for intialization and calibration
def draw_circle(q1,q2,q3):   #画图函数
    screen = pygame.display.set_mode(size,pygame.FULLSCREEN)
    screen.fill(WHITE) # 填充屏幕
    pygame.draw.circle(screen, BLACK, q1, 10, 0) 
    pygame.draw.circle(screen, BLACK, q2, 10, 0) 
    pygame.draw.circle(screen, BLACK, q3, 10, 0) 

    pygame.display.flip()

def draw_grids(q):   #测试函数
    screen = pygame.display.set_mode(size,pygame.FULLSCREEN)
    screen.fill(WHITE) # 填充屏幕
    divide=10
    for i in range(1,divide):   #horizon
        pygame.draw.line(screen, BLACK, (0, height*i/divide), (width,height*i/divide), 2)

    for i in range(1,divide):   #vertical
        pygame.draw.line(screen, BLACK, (width*i/divide,0), (width*i/divide, height), 2)
    pygame.draw.circle(screen, RED, q[0], 10, 0) 
    pygame.draw.circle(screen, RED, q[1], 10, 0) 
    pygame.draw.circle(screen, RED, q[2], 10, 0) 
    pygame.draw.circle(screen, RED, q[3], 10, 0) 
    pygame.draw.circle(screen, RED, q[4], 10, 0) 
    pygame.display.flip()

def get_pos():   #return the position where the mouse clicks 
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
                    #print(mouse_position)
                    return mouse_position

def Get_picture(i):    #get, save and visualize a picture from camera
    while (1):
        ret, frame = cap.read()
        flag=0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
                raise KeyboardInterrupt  # 退出游戏
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                
                pic_size = frame.shape
                #print(frame.shape)
                cv2.imwrite('./' + str(i) + '.jpg', frame)
                flag=1
        if flag==1:
            break

    print("Get pic!!")
    #print(pic_size)
    screen = pygame.display.set_mode(pic_size, pygame.FULLSCREEN)
    screen.fill(WHITE)
    space = pygame.image.load('./' + str(i) + '.jpg', frame)
    screen.blit(space, (0, 0))
    # 3.刷新游戏窗口
    pygame.display.update()


def initialization():
    q1=(540,120)
    q2=(140,630)
    q3=(840,630)
    draw_circle(q1,q2,q3)

    Get_picture(0)   #0 is the number of picture
    p1=get_pos()
    p2=get_pos()
    p3=get_pos()
    calibration(q1,q2,q3,p1,p2,p3)    # calculate the calbration cofficient


    #test the calibratiob
    #左上，右上，右下，左下，中间
    test_q= ((0,0), (1280,0), (1280,800), (0,800), (640,400))
    error = 15
    draw_grids(test_q)
    Get_picture(1)
    for num in range(0, 5):
        p = convert_position(get_pos())
        if abs(p[1]-test[num][1]) > error or abs(p[0]-test[num][0]) > error:
            print("Calibration Error: Fail to acquire an accurate calibration!")
            sys.exit()
            raise KeyboardInterrupt  # 退出游戏
    screen = pygame.display.set_mode(size,pygame.FULLSCREEN)
    screen.fill(WHITE) # 填充屏幕
    pygame.display.flip()
    
initialization()
connect().start()
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
        cap.release()
        end = 1
        print('Goodbye')
