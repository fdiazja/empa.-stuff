#!/usr/bin/env python
import os,time
import numpy as np
from scipy.stats import linregress
f = lambda a,b,x: a*x+b
bobbin = 2
shift = 0.000103
sh = (201,328,320)
from scipy import ndimage as ndi
v = np.copy(np.memmap('data/Sp%d_calib_'%bobbin+'%dx%dx%d.raw'%sh[::-1],dtype=np.float32,shape=sh,mode='r'))+shift
folder = 'log/bobbin_%d'%bobbin
if not os.path.exists(folder): os.mkdir(folder)
os.chdir(folder)

"""
imb = np.array(v>0.).astype(np.uint8)
t0 = time.time()
d = ndi.distance_transform_edt(imb)
print time.time()-t0
d.astype(np.float32).tofile('d_%dx%dx%d.raw'%d.shape[::-1])
import sys; sys.exit()
"""

d = np.memmap('d_%dx%dx%d.raw'%sh[::-1],dtype=np.float32,shape=sh)
markers = ndi.label(d>0)
markers0 = markers[0].astype(np.uint8)
markers0[markers0>5] = 0
markers0 = ndi.morphology.binary_closing(markers0,structure=np.ones((1,3,3)),iterations=2)
markers0 = ndi.morphology.binary_erosion(markers0,structure=np.ones((1,3,3)),iterations=2)
markers0 = ndi.morphology.binary_erosion(markers0,structure=np.ones((1,3,3)),iterations=3)
m = ndi.label(markers0)[0].astype(np.uint8)
v[m==0] = 0
v.astype(np.float32).tofile('v_dx%dx%d.raw'%v.shape[::-1])
m[v==0] = 6
m.tofile('m_%dx%dx%d.raw'%m.shape[::-1])

dens = [ [d[m==i].mean() for i in {1:[4,2,1,3],2:[3,4,2,1]}[bobbin]],[0.25967,0.45659,0.49753,0.68773] ]
print dens

slope,intercept,r_value,p_value,std_err = linregress(dens[0],dens[1])
print slope,intercept
R2 = r_value**2
x = np.linspace(dens[0][0],dens[0][-1],100)

import matplotlib.pyplot as plt
plt.ion()
plt.clf()  
plt.plot(dens[0],dens[1],marker='o')
plt.xlabel('gv')
plt.ylabel(r'$\rho$ [$g/cm^{3}$]')
plt.title('Bobbin %d'%(bobbin))
if bobbin==2: plt.text(0.002,0.62,r'$\rho$ = %.4f gv + %.4f'%(slope,intercept))
else: plt.text(0.0006,0.62,r'$\rho$ = %.4f gv + %.4f'%(slope,intercept))
plt.plot(x,f(slope,intercept,x),label='fit',linewidth=1,linestyle='--')
plt.xlim()
plt.grid()	  
plt.show()
plt.savefig('log/fit_%d.png'%(bobbin))
raw_input()

plt.clf()
import matplotlib.cm as cm
sh = (2004,2004)
d = np.copy(np.memmap('data/AVG_Sp2.raw',dtype=np.float32,shape=sh)) + shift
#sh = (201,2004,2004)
#d = np.copy(np.memmap('data/Sp2_2004x2004x201.raw',dtype=np.float32,shape=sh)[100]) + shift[bobbin]
fig,ax = plt.subplots()
for i in ['gist_ncar']:#,'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr','RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral','seismic']:
  print i
  plt.clf()
  im = plt.imshow(np.clip(f(slope,intercept,d),0.15,0.7),cmap=getattr(cm,i))#interpolation='nearest',)
  cbar = fig.colorbar(im,orientation='vertical')
  plt.savefig('log/map_%d.png'%(bobbin))  
  raw_input()
