import numpy as np
import os,sys
from datetime import datetime

def read_gpx_file(fn):
    pass

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