#import Jetson.GPIO
import time
import serial

with serial.Serial("/dev/ttyACM0", 9600,  timeout=10) as ser:
    while True:
        led_on = input("LED ON?")[0]
        if led_on in "yY":
            ser.write(bytes("Yes\n", "utf-8"))
        elif led_on in "nN":
            ser.write(bytes("No\n", "utf-8"))


#header = "/dev
#ser = serial.Serial(header,115200,timeout=1)
#five_sec = time.time()+5
#while time.time() < five_sec:
#    ser.write(b"1")
#print("done")

