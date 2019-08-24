from upysh import *
import network
import socket
import utime
from machine import Pin

p=Pin(2,mode=Pin.IN)
ap_if = network.WLAN(network.AP_IF)
ap_if.active(False)
ap_if.active(True)
ap_if.config(essid='ESP8266',password='12345678')

print(ap_if.ifconfig()[0])
addr = socket.getaddrinfo('0.0.0.0',80)[0][-1]
state = 0 #0-stop，1-move
connect=0
s = socket.socket()
s.bind(addr)
s.listen(1)

try:         
    print("连接中.....")
    conn, addr = s.accept()
    print('client connected from', addr)
    while True:
        request = conn.recv(1024)
        conn.send(b'1') 
        
except KeyboardInterrupt:
    conn.close()