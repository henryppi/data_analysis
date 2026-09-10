import numpy as np
import os,sys
from datetime import datetime

def read_file(filename,last_n_lines):
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    with open(filename) as f:
        content = f.readlines()[-last_n_lines:]
    n = len(content)
    data = []
    t = []
    for i in range(n):
        string = content[i]
        string = string[:string.rfind("\n")]
        string = string.split(',')
        time_string = string[0]
        datetime_object = datetime.strptime(time_string, time_format)
        t.append(datetime_object)
        nRow = len(string)
        row = []
        for col in range(nRow)[1:]:
            if string[col]=='None':
                row.append(np.nan)
            else:
                row.append(float(string[col]))
        data.append(row)
    return np.array(t),np.array(data)

def moving_average(data,win=3):
    return np.convolve(data, np.ones(win), 'valid') / win

def iir_low_pass(t,y,cutoff_freq):
    n = len(y)
    dt = np.mean(np.diff(t))
    rc = 1.0/(2*np.pi*cutoff_freq)
    alpha = dt /( rc + dt )
    print('dt = ',dt, ' sample_freq = ',1/dt, ' rc = ',rc, ' alpha = ', alpha)
    y_filtered = np.zeros(n)
    y_filtered[0] = y[0]
    for i in range(1,n):
        y_filtered[i] = alpha * y[i] + (1-alpha)*y_filtered[i-1]
    return y_filtered 