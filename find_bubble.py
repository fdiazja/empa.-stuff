import os
os.system('del /F /Q log\*')
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage import filters
import scipy.ndimage as ni
import skimage.measure as skm
import scipy.ndimage as ni
import scipy.ndimage.morphology as nim

sh = (Nz,Ny,Nx) = (4615,1024,1024)
a = np.memmap('1wcgt1_HFTR_%dx%dx%d.raw'%sh[::-1],dtype=np.uint16,offset=0,shape=sh,mode='r')
im = a[500].astype(np.float32)
im.astype(np.float32).tofile('log/orig_%dx%d.raw'%im.shape[::-1])
im = np.gradient(im)
"""
for i in np.arange(1, 51, 2):
    im2 = im/threshold_local(im,block_size=i,method='median',offset=0)
    im2[np.where(im2 > 5)] = 0
    im2.astype(np.float32).tofile('log/th_%dx%d'%im2.shape[::-1] + '_%d.raw'%i)
    im3 = im2 > 1.02
    im3 = im * im3
    im3.astype(np.float32).tofile('log/binary_%dx%d'%im3.shape[::-1] + '_%d.raw'%i)
    labeled = skm.label(im3, neighbors=8)
    labeled.astype(np.float32).tofile('log/labeled_%dx%d'%labeled.shape[::-1] + '_%d.raw'%i)
"""	
im2 = im / threshold_local(im, block_size=27, method='median', offset=0)
im2[np.where(im2 > 3)] = 0
iter = np.arange(1, 1.1, 0.001)
for i, j in zip(iter, range(len(iter))):
  #im2.astype(np.float32).tofile('log/th_%dx%d.raw'%im2.shape[::-1])
  im3 = im2 > i
  #im3.astype(np.float32).tofile('log/binary_%dx%d.raw'%im3.shape[::-1])
  labeled = skm.label(im3, neighbors=4)

  for region in skm.regionprops(labeled):
    if region.area < 50: labeled[labeled == region.label] = 0

  #labeled.astype(np.float32).tofile('log/labeled_%dx%d'%labeled.shape[::-1] + '_%d.raw'%i)
  im4 = labeled != 0
  im4 = ni.binary_closing(im4, iterations=2)
  im4 = nim.binary_fill_holes(im4)
  im4 = im4 * im
  im4.astype(np.float32).tofile('log/binary_better_%dx%d'%im4.shape[::-1] + '_%d.raw'%j)