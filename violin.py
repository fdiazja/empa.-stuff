import os
os.system('del /F /Q log\*')
import numpy as np
import skimage.measure as skm
from skimage.filters import threshold_otsu
import scipy.ndimage as ni
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

sh = (Nz, Ny, Nx) = (2446, 478, 1166)
init = 0
fin = 2446
data = np.memmap('ganze-geige-oben-skaliert-mit-0_72_%dx%dx%d.raw'%sh[::-1], dtype=np.uint16, offset=0, shape=sh, mode='r').astype(np.float32)
stack = data[init:fin]
thresh = threshold_otsu(stack)
binary = stack < thresh
binary = ni.binary_fill_holes(binary)
binary.astype(np.float32).tofile('log_binary/binary_%dx%dx%d.raw'%binary.shape[::-1])

"""
labeled = skm.label(binary)
labeled[np.where(labeled > 2)] = 0
labeled.astype(np.float32).tofile('log_labeled/labeled_%dx%dx%d.raw'%labeled.shape[::-1])
"""

for i in range(fin - init):
  labeled = skm.label(binary[i])
  labeled[np.where(labeled > 2)] = 0
  for r in skm.regionprops(labeled):
    if r.area < 10e3: labeled[np.where(labeled == r.label)] = 0 
  labeled.astype(np.float32).tofile('log_labeled/labeled_%dx%d'%labeled.shape[::-1] + '_%d.raw'%i)
