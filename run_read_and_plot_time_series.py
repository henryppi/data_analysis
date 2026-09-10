import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sources.data_analysis_tools import read_file


fn = './data_files/data_20250904_104729.dat'

n_lines = 33000
t,data = read_file(fn,n_lines)

a = 1000
b = 10000

mean = np.mean(data[a:b,2])
std = np.std(data[a:b,2])
print(mean,std)

data[:,2] += -mean

weight_ref = 99.5

c = 14000
d = 19000

mean2 = np.mean(data[c:d,2])
print(mean2)

data[:,2] *= weight_ref/mean2

std2 = np.std(data[a:b,2])
std3 = np.std(data[c:d,2])

print(std2,std3)

n_avg = 240
weights = np.ones(n_avg)/n_avg
avg = np.convolve(data[:,2],weights,mode='valid')



plt.plot(data[:,2],'.k',markersize=1)
plt.plot(avg[(n_avg-1):],'-r',lw=2)
plt.show()