import threading
import time
import socket

exitFlag = 0
count=1
class MainThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        count+=1


class TcpThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print ("Starting TCP server")
        TCP_IP = '127.0.0.1'
        TCP_PORT = 5005
        BUFFER_SIZE = 64  # Normally 1024, but we want fast response

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        conn, addr = s.accept()

        print ('Connection address:', addr)
        while True:
            data = conn.recv(BUFFER_SIZE)
            if not data: 
                print("no data")
                continue
            print ("received data:", data)
            conn.send(data)  # echo
        print ("Exiting TCP")


# Create new threads
tcpThread = TcpThread()

tcpThread.start()
# mainThread.start()
tcpThread.join()
while True:
    print("a")
# mainThread.join()
print ("Exiting Main Thread")