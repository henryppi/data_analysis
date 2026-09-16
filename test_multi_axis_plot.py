import numpy as np
from sources.data_analysis_tools import *

import matplotlib.pyplot as plt
plt.ion()

fn = './data_files/data_20260916_114249.dat'

t,data_ = read_file(fn,1000)
n = data_.shape[0]
ndata = data_.shape[1]

t = t.astype('datetime64[us]').astype(float)/1e6
t += -t[0]

data = np.zeros([n,ndata+1],float)
data[:,0] = t
data[:,1:] = data_

moni = Monitor(ndata,n)
moni.set_data(data)
moni.update()

plt.ioff()
plt.show()
