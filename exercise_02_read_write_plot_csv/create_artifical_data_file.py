import numpy as np

filename = 'measurement.csv'

tStart = 0
tEnd = 100
nTimeSteps = 1000

y0 = 1
mass = 0.5
damp = 0.05
spring = 2.0
noise_amp = 0.05

omega0 = np.sqrt(mass/spring)

t = np.linspace(tStart,tEnd,nTimeSteps)
y_noise = noise_amp*(np.random.randn(nTimeSteps) -0.5)
y = y0*np.cos(omega0*t)*np.exp(-damp*t) + y_noise

data = np.zeros([nTimeSteps,2],float)
data[:,0] = t
data[:,1] = y

np.savetxt(filename,data,delimiter=",")
