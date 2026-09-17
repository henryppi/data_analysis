import numpy as np
import time
from sources.data_analysis_tools import *
import matplotlib.pyplot as plt


# vert,elem = make_cube(2,1,0.5)
vert,elem = make_ship(2,1,0.5)

cad = cad_object()
cad.set_geom(vert,elem)
cad.set_center_bbox()


moni3D = Monitor3D(cad)


moni3D.set_pose(0,0,0,0*np.pi/180,0,0)
moni3D.update()