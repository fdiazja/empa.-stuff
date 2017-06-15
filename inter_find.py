import os
os.system('del /F /Q log\*')
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as mg  
import scipy.ndimage.measurements as mm 
from scipy import ndimage as ni
from scipy.interpolate import interp1d
from scipy import ndimage as ndi
from scipy.ndimage import label,filters
from skimage.filters import rank
from skimage.morphology import watershed,disk
from skimage import filters

sh = (Nz,Ny,Nx) = (4615,1024,1024)
a = np.memmap('1wcgt1_HFTR_%dx%dx%d.raw'%sh[::-1],dtype=np.uint16,offset=0,shape=sh,mode='r')
init = 50
to = 70
iter = to - init
step = 50
b = np.copy(a[init:to])
#b.astype(np.float32).tofile('log/b_%dx%dx%d.raw'%b.shape[::-1])

"""
#Interface with labeling
vec = np.arange(init + step, to, step)
h_mean = np.zeros(vec.shape)
for i in range(iter):
  b_b = b[i] > 40000
  b_b = b_b[::, 200:(b_b.shape[0] - 100)]
  c = mg.binary_dilation(mg.binary_closing(b_b))
  labeled, nl = ni.label(c)
  labeled[labeled != 1] = 0 
  inter_line = ni.filters.sobel(labeled)
  inter_line[inter_line != 0] = 1
  #inter_line.astype(np.float32).tofile('log/inter_line_%dx%d'%b_b.shape[::-1] + '_%04d.raw'%i)
  indy_dummy =  np.argmax(inter_line,axis=0)
  y = np.where(indy_dummy!=0)
  y = np.asarray(y)[0]
  func = interp1d(y,np.take(indy_dummy,y))
  indx = np.arange(np.amin(y), np.amax(y))
  indy = func(indx)
  indy = np.round(indy).astype(np.int)
  marked = b[i]
  marked[indy, indx + 200] = 0
  marked.astype(np.float32).tofile('log/marked_%dx%d'%marked.shape[::-1] + '_%04d.raw'%i)
  if i%step == 0 and i != 0:
    h_mean[(i / step) - 1] =  np.mean((b[i].shape[0] - indy))

plt.plot(vec, h_mean)
plt.ylabel('Height (pixels)')
plt.xlabel('Image number')
plt.grid(True)
plt.show()
"""

