import socket
import numpy as np

UDP_IP = "10.152.246.117"
UDP_PORT = 9090
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

prediction = np.ones(18)
# prediction = np.hstack((count,prediction))
# prediction.astype(np.float16)
print(prediction.shape)

#! send ucp
MESSAGE = prediction.tostring()
# print(sys.getsizeof(MESSAGE))
print(np.fromstring(MESSAGE,prediction.dtype).shape)
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))


MESSAGE = bytes("Hello, World!",'utf8')

print ("UDP target IP:", UDP_IP)
print ("UDP target port:", UDP_PORT)
print ("message:", MESSAGE)

sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))