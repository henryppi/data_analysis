import sys, time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sources.data_analysis_tools import read_file


def main(fn,last_n_lines):
    t_wait = 0.0 
    plt.ion()
    fig, ax = plt.subplots(1, 1, sharex=True)
    t,data = read_file(fn,last_n_lines)
    line, = ax.plot(t, data[:,2]) 
    
    while True:
        t,data = read_file(fn,last_n_lines)
        line.set_xdata(t)
        line.set_ydata(data[:,2])
        ax.set_xlim([t[0],t[-1]])
        ax.set_ylim([np.min(data[:,2])-1,np.max(data[:,2])+1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.draw()
        plt.pause(0.01)
        time.sleep(t_wait)

if __name__ == "__main__":
    if len(sys.argv)==3:
        fn = sys.argv[1]
        last_n_lines = int(sys.argv[2])
        main(fn,last_n_lines)
    else:
        print('wrong arguments\nuse "'+sys.argv[0]+' filename nlines"')
