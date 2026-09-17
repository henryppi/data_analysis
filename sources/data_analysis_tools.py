import numpy as np
import os,sys
from datetime import datetime
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

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

    
def make_cube(L,B,H):
    vert = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]],float)
    vert[:,0] *=L
    vert[:,1] *=B
    vert[:,2] *=H
   
    elem = [[0,1,2,3],[0,1,6,7],[1,2,5,6],[2,3,4,5],[3,0,7,4],[4,5,6,7]]

    return vert.T,elem

def make_ship(L,B,H):
    B2 = B/2
    f1 = 0.8
    f2 = 0.6
    f3 = 0.8
    f4 = 0.6

    vert = np.array([[0,0,0],\
                     [0,0,H],\
                     [0,B2,H],\
                     [0,B2*f1,H*(1-f1)],\
                     [L*f2,0,H],\
                     [L*f2,B2,H],\
                     [L*f2,B2*f1,H*(1-f1)],\
                     [L*f2,0,0],\
                     [,,],\
                    #  [,,],\
    ])

    elem = [[0,1,2,3],\
            [1,4,5,2],\
            [2,5,6,3],\
            [3,6,7,0],\
    ]
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
        
    def set_pose(self,x_vec):
        self.x_vec = x_vec
        
    def apply_pose(self):
        self.vert = transformation(self.vert0, self.x_vec, self.M)
        
    def get(self):
        return self.vert, self.elem
    
    def get_bbox(self):
        return [np.min(self.vert0[:,0]),
                np.max(self.vert0[:,0]),
                np.min(self.vert0[:,1]),
                np.max(self.vert0[:,1]),
                np.min(self.vert0[:,2]),
                np.max(self.vert0[:,2])]

    def set_center(self,x,y,z):
        self.M[:,0] = [x,y,z]

    def set_center_bbox(self):
        bbox = self.get_bbox()
        self.set_center((bbox[0]+bbox[1])*0.5,
                        (bbox[2]+bbox[3])*0.5,
                        (bbox[4]+bbox[5])*0.5)

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

        if self.nrow>1:
            self.fig, self.axs = plt.subplots(self.nrow,1,sharex=True,figsize=(8,8),facecolor='w',frameon=False)
            self.axs = self.axs.flatten()
        else:
            self.fig, axs = plt.subplots(self.nrow,1,sharex=True,figsize=(8,8),facecolor='w',frameon=False)
            self.axs = [axs]
        
        self.fig.patch.set_facecolor('none') 
        
        for i,ax in enumerate(self.axs):
            ax.patch.set_facecolor('none')
            # ax.twinx()

        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(1000/dpi,600/dpi)
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
        # print('update ',self.data[-1,0])

        for i in range(self.nrow):
            self.line_list[i].set_data( self.data[-self.ndata:,0], self.data[-self.ndata:,i+1])
        
        for i in range(self.nrow):
            self.axs[i].relim()
            self.axs[i].autoscale_view()
            self.axs[i].set_xlim([self.data[-self.ndata,0],self.data[-1,0]])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Monitor3D:
    def __init__(self,cad):
        self.flag_body = True
        self.flag_triad = True
        self.flag_trace = True

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # self.fig, self.ax = plt.subplots(1,1,facecolor='w',frameon=False,projection='3d')
        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(600/dpi,600/dpi)
        self.fig.canvas.manager.window.move(900, 100)
        self.ax.axis('equal')
        self.cad = cad
        self.initialize()

    def initialize(self):
        if self.flag_body:
            colors = np.linspace(0,100,6)
            cad_poly3d = self.cad.get_poly3d()
            self.patch3D = Poly3DCollection(cad_poly3d, cmap=matplotlib.cm.jet, edgecolors='k',  linewidths=2, alpha=0.50)
            self.patch3D.set_array(colors)
            self.ax.add_collection3d(self.patch3D)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.draw()

    def set_pose(self,x,y,z,phi,psi,chi):
        self.x_vec = np.array([[x,y,z,phi,psi,chi]],float).T
        self.cad.set_pose(self.x_vec)
        
    def update(self):
        
        self.cad.apply_pose()
        
        self.patch3D.set_verts(self.cad.get_poly3d())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.axis('equal')
        plt.draw()
        plt.show()


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