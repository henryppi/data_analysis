import numpy as np
import time
from sources.data_analysis_tools import *
import matplotlib.pyplot as plt
plt.ion()
nrow = 6
ndata = 1000
dt = 0.01
t = 0.0

data = np.zeros([ndata,nrow+1],float)

moni = Monitor(nrow,ndata)

try:
    while True:
        rng = np.random.normal(loc=0.0,scale=0.1,size=nrow)
        data_row = np.zeros(nrow+1)
        data_row[0] = dt
        data_row[1:] = rng
        data = np.append(data,np.array([data[-1,:]+data_row]),axis=0)
        moni.set_data(data)
        moni.update()
        time.sleep(dt)
        t +=dt
except KeyboardInterrupt:
    pass
