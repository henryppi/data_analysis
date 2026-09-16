import numpy as np
import os,sys
from datetime import datetime
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

def read_file(filename,last_n_lines=-1):
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    with open(filename) as f:
        if last_n_lines==-1:
            content = f.readlines()[1:]
        else:
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

def make_patch_cube(L,B,H):
    vert = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]],float)
    vert[:,0] *=L
    vert[:,1] *=B
    vert[:,2] *=H
    
    x  = vert[:,0]
    y  = vert[:,1]
    z  = vert[:,2]
    
    elem = [[0,1,2,3],[0,1,6,7],[4,5,6,7]]
    
    tupleList = list(zip(x, y, z))

    poly3d = [[tupleList[elem[ix][iy]] for iy in range(len(elem[0]))] for ix in range(len(elem))]

    return poly3d
    
def make_cube(L,B,H):
    vert = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]],float)
    vert[:,0] *=L
    vert[:,1] *=B
    vert[:,2] *=H
   
    elem = [[0,1,2,3],[0,1,6,7],[1,2,5,6],[2,3,4,5],[3,0,7,4],[4,5,6,7]]

    return vert.T,elem

def transformation(points_old,x_vec,M):
    n = points_old.shape[1]
    points = np.zeros([3,n])
    phi=x_vec[3,0]
    psi=x_vec[4,0]
    chi=x_vec[5,0]

    Trotx = np.matrix([[1,0,0],\
             [0, np.cos(phi),-np.sin(phi)],\
             [0, np.sin(phi), np.cos(phi)]])
    Troty = np.matrix([[np.cos(psi), 0, np.sin(psi)],\
                    [0, 1, 0],\
                    [-np.sin(psi), 0, np.cos(psi)]])

    Trotz = np.matrix([[np.cos(chi), -np.sin(chi), 0],\
                    [np.sin(chi), np.cos(chi), 0],\
                    [0, 0, 1]])
    ROT=Trotz*Troty*Trotx

    for i in range(n):
        points[:,i] = (ROT*(np.array([points_old[:,i]]).T-M) + M + np.array([x_vec[0:3,0]]).T).flatten()
    return points

class cad_object:
    def __init__(self):
        self.vert0 = []
        self.elem0 = []
        self.vert = []
        self.elem = []
        self.vert = []
        self.x_vec = np.array([[0,0,0,0,0,0]],float).T
        self.M = np.array([[0,0,0]]).T
    
    def set_geom(self,points,elements):
        self.vert0 = points
        self.elem0 = elements
        self.vert = points
        self.elem = elements
        
    def set_pose(self,x_vec,M):
        self.x_vec = x_vec
        self.M = M
        
    def apply_pose(self):
        self.vert = transformation(self.vert0, self.x_vec, self.M)
        
    def get(self):
        return self.vert, self.elem
        
    def get_poly3d(self):
        x  = self.vert[0,:]
        y  = self.vert[1,:]
        z  = self.vert[2,:]
        tupleList = list(zip(x, y, z))
        poly3d = [[tupleList[self.elem[ix][iy]] for iy in range(len(self.elem[0]))] for ix in range(len(self.elem))]
        return poly3d

class Monitor:
    def __init__(self,nrow=1,ndata=10):
        self.nrow = nrow
        self.ndata = ndata

        self.data = np.zeros([ndata,nrow+1],float)
        self.data[:,0] = np.linspace(0,1,self.ndata)

        self.fig, self.axs = plt.subplots(self.nrow,1,sharex=True,figsize=(8,8),facecolor='w',frameon=False)

        self.axs = self.axs.flatten()
        self.fig.patch.set_facecolor('none') 
        for i,ax in enumerate(self.axs):
            ax.patch.set_facecolor('none')
        
        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(800/dpi,600/dpi)
        self.fig.canvas.manager.window.move(200, 100)

        self.line_list = []
        for i in range(nrow):
            line_tmp, = self.axs[i].plot(self.data[-self.ndata:,0], self.data[-self.ndata:,i+1],'-k',lw=1)
            self.line_list.append(line_tmp)

        for i in range(self.nrow):
            self.axs[i].set_autoscaley_on(True)
            self.axs[i].set_xlim([self.data[0,0],self.data[-1,0]])
    
    def set_data(self,data):
        self.data = data
    
    def update(self):
        print('update ',self.data[-1,0])

        for i in range(self.nrow):
            self.line_list[i].set_data( self.data[-self.ndata:,0], self.data[-self.ndata:,i+1])
        
        for i in range(self.nrow):
            self.axs[i].relim()
            self.axs[i].autoscale_view()
            self.axs[i].set_xlim([self.data[-self.ndata,0],self.data[-1,0]])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def moving_average(data,win=3):
    return np.convolve(data, np.ones(win), 'valid') / win

def iir_low_pass(t,y,cutoff_freq):
    n = len(y)
    dt = np.mean(np.diff(t))
    rc = 1.0/(2*np.pi*cutoff_freq)
    alpha = dt /( rc + dt )
    print('dt = ',dt, ' sample_freq = ',1/dt, ' rc = ',rc, ' alpha = ', alpha)
    y_filtered = np.zeros(n)
    y_filtered[0] = y[0]
    for i in range(1,n):
        y_filtered[i] = alpha * y[i] + (1-alpha)*y_filtered[i-1]
    return y_filtered 