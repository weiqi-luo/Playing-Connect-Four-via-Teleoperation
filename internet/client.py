#!/usr/bin/env python

import socket
import time
import numpy as np

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024
MESSAGE = "Hello, World!"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
count = 0
while True:
    count += 1
    MESSAGE = str(count)
    s.send(MESSAGE)
    data = s.recv(BUFFER_SIZE)
    data = np.fromstring(data, dtype=np.float16) 
    print "received data:", data

s.close()
