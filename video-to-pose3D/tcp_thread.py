from threading import Thread, Event
from time import sleep
import time
import socket,sys
import numpy as np
import datetime
# event = Event()

def tcpThread(var, event): 
    dt = datetime.datetime
    try:       
        print ("Starting TCP server")
        TCP_IP = '192.168.1.43'
        TCP_PORT = dt.now().hour*100 + dt.now().minute + 1024
        BUFFER_SIZE = 64  # Normally 1024, but we want fast response

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        conn, addr = s.accept()

        print ('Connection address:', addr)
        while True:
            data = conn.recv(BUFFER_SIZE)
            if not data: 
                print (time.ctime(time.time()))
                continue
            print ("received data:", data)
            if var[0] is not None:
                conn.send(var[0])  # echo
            # sys.exit()
            
        print ("Exiting TCP")
    finally:
        event.set()
        s.close()

if __name__ == "__main__":    
    my_var = [0]
    t = Thread(target=tcpThread, args=(my_var, ))
    t.start()
    while True:
        try:
            print(my_var)
            # sleep(1)
        except KeyboardInterrupt:
            # event.set()
            break
    t.join()
    print(my_var)