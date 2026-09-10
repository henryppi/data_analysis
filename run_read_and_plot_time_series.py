import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sources.data_analysis_tools import (read_file,
                                         moving_average,
                                         iir_low_pass)


fn = './data_files/data_20250904_104729.dat'

n_lines = 33000
t,data = read_file(fn,n_lines)

data = data[:,2]

t = t.astype('datetime64[us]').astype(float)/1e6
t += -t[0]

win = 100
data_avg = moving_average(data,win=win)


cutoff_freq = 0.5
data_low = iir_low_pass(t,data,cutoff_freq)

plt.plot(t,data,'-k',markersize=1,lw=0.5)
plt.plot(t[(win-1):],data_avg,color='r',marker='+',linestyle='None',lw=1,markersize=2)
plt.plot(t,data_low,'og',lw=2,markersize=2)
plt.show()