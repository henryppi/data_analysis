import sys
import serial

def monitor_usb(usb_port,speed):
    ser = serial.Serial(usb_port,int(speed))
    while True:
        line = ser.readline()
        print(line)

if __name__ == "__main__":
    if len(sys.argv)==3:
        usb_port = sys.argv[1]
        speed = sys.argv[2]
        monitor_usb(usb_port,speed)
    else:
        print('wrong arguments\nuse "'+sys.argv[0]+' usb-port baud-rate"')