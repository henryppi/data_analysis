import numpy as np
import time
from sources.data_analysis_tools import *
import matplotlib.pyplot as plt
plt.ion()
nrow = 6
ndata = 200
dt = 0.025
t_wait = dt
t=0.0
data = np.zeros([ndata,nrow+1],float)

moni = Monitor(nrow,ndata)

t_1 = time.time()

try:
    while True:
        t_0 = time.time()
        
        rng = np.random.normal(loc=0.0,scale=0.1,size=nrow)
        data_row = np.zeros(nrow+1)
        data_row[0] = dt
        data_row[1:] = rng
        data = np.append(data,np.array([data[-1,:]+data_row]),axis=0)
        moni.set_data(data)
        moni.update()
        t_wait = dt -(time.time()-t_0)
        print(t_wait)
        if t_wait > 0.0:
            time.sleep(t_wait)

        t +=dt
except KeyboardInterrupt:
    pass
