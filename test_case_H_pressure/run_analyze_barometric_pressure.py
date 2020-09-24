import numpy as np
import matplotlib.pyplot as plt
import requests


url = 'https://raw.githubusercontent.com/henryppi/data_analysis/master/test_case_H_pressure/data_pressure/RawData1.csv'

response = requests.get(url)
text=response.iter_lines(decode_unicode='utf-8')
data = np.genfromtxt(text,delimiter=',',skip_header=1)

plt.plot(data[:,0],data[:,1])
plt.show()
