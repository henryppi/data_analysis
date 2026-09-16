import sys, time
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from sources.data_analysis_tools import read_file, cad_object, make_cube
   
def main(fn,last_n_lines):
    t_wait = 0.0 # plot update interval
    plt.ion()
    
    vert,elem = make_cube(2,1,0.5)
    
    cad = cad_object()
    cad.set_geom(vert,elem)
    cad_poly3d = cad.get_poly3d()
    
    x_vec = np.array([[0,0,0,0,0,0]],float).T
    M = np.array([[1,0.5,0.25]]).T
    
    colors = np.linspace(0,100,6)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    patch3D = Poly3DCollection(cad_poly3d, cmap=matplotlib.cm.jet, edgecolors='k',  linewidths=2, alpha=0.90) #facecolors='w',
    patch3D.set_array(colors)
    ax.add_collection3d(patch3D)

    ax.set_xlim([-1,2])
    ax.set_ylim([-1,2])
    ax.set_zlim([-1,2])

    t,data = read_file(fn,last_n_lines)
        
    while True:
        t,data = read_file(fn,last_n_lines)
        
        x_vec[3,0] = -np.mean(data[:,2])*np.pi/180
        x_vec[4,0] = np.mean(data[:,3])*np.pi/180
        x_vec[5,0] = -np.mean(data[:,1])*np.pi/180+np.pi/2
        
        cad.set_pose(x_vec,M)
        cad.apply_pose()
        patch3D.set_verts(cad.get_poly3d())

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
