import numpy as np
import matplotlib.pyplot as plt

filename = 'measurement.csv'

data = np.loadtxt(filename,delimiter=',')

plt.plot(data[:,0],data[:,1])
plt.show()
