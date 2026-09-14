import serial  # use pip install pyserial NOT the module 'serial' 
from datetime import datetime, timedelta

# ser = serial.Serial('/dev/tty.usbserial-0001',115200, timeout=1)
# ser = serial.Serial('/dev/ttyUSB0',115200) # linux
ser = serial.Serial('/dev/cu.usbserial-0001',115200) # macos
# serBT = serial.Serial('/dev/tty.esp32send-ESP32SPP',115200, timeout=1)

time_now = datetime.now()
fn_datetime = time_now.strftime("%Y%m%d_%H%M%S")
filename = './data_files/data_'+fn_datetime+'.dat'
delimiter = ','
time_format = "%Y-%m-%d %H:%M:%S.%f"
header = '''# time,counter,data\n'''

with open(filename, "w") as myfile:
    myfile.write(header)

counter = 1

while True:
    line = ser.readline()
    try:
        line = line.decode('utf-8')
    except:
        print("line.decode('utf-8') failed")
    
    if line[0]=='#':
        print(line)
        continue
    
#     line = line.decode('utf-8')
    line = line.replace('\n','').replace('\r','').replace(' ','')
    line = line.split(delimiter)
    
       
    # print(line)
    time_now = datetime.now()
    time_string = time_now.strftime(time_format)   

    line_string = "{},{}".format(time_string,counter)
    
    n_col = len(line)
    for i in range(n_col):
        try:
            tmp_string = ",{}".format(float(line[i]))
        except:
            print('failed string conversion')
            tmp_string = ", "

        # line_string += ",{}".format(float(line[i]))
        line_string += tmp_string
#     data = int(line[1])
    line_string += "\n"
    with open(filename, "a") as myfile:
            myfile.write(line_string)
            
    counter += 1
    if counter%100==0:
        print('counter = ',counter)