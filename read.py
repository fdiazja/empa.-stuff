import os,cv2
os.system('del /F /Q log\*')
import numpy as np
from scipy.ndimage import label,filters
from skimage.morphology import watershed,disk
from skimage.filters import rank

sh = (Nz,Ny,Nx) = (4615,1024,1024)
a = np.memmap('1wcgt1_HFTR_%dx%dx%d.raw'%sh[::-1],dtype=np.uint16,shape=sh,mode='r')
b = a[190]
imb = np.array(b<38000)
from scipy import ndimage as ndi
dist = ndi.distance_transform_edt(imb)
dist.astype(np.float32).tofile('log/dist_%dx%dx.raw'%dist.shape[::-1])
b.astype(np.uint16).tofile('b_%dx%d.raw'%b.shape[::-1])
import sys; sys.exit()

"""
#watershed
d = rank.median(b,disk(2))
d.astype(np.uint16).tofile('log/d_%dx%d.raw'%d.shape[::-1])

d = np.memmap('d_1024x1024.raw',dtype=np.uint16,shape=(1024,1024))
markers = rank.gradient(d,disk(4))<10
markers = label(markers)[0]
g = rank.gradient(d,disk(1))
labels = watershed(g,markers)
g.astype(np.uint16).tofile('log/g_%dx%d.raw'%g.shape[::-1])
b = np.copy(np.memmap('b_1024x1024.raw',dtype=np.uint16,shape=(1024,1024)))
"""
g = np.memmap('g_1024x1024.raw',dtype=np.uint16,shape=(1024,1024))

"""
d = np.memmap('d_1024x1024.raw',dtype=np.uint16,shape=(1024,1024)).astype(np.ubyte)
e = cv2.Canny(d,100,200)
e.astype(np.uint16).tofile('log/e_%dx%d.raw'%e.shape[::-1])

circles = cv2.HoughCircles(g,cv2.cv.CV_HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
print circles
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(g,(i[0],i[1]),i[2],(0,255,0),2)
cv2.imshow('detected circles',g)
g.tofile('test.raw')
raw_input()
"""
#g = cv2.adaptiveThreshold(g.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
g = (2**8*np.clip(g,0,10000.)/10000.).astype(np.uint8)
g.tofile('log/g_%dx%d.raw'%g.shape[::-1])
h = cv2.Canny(g,30,200)
h.astype(np.uint8).tofile('log/h_%dx%d.raw'%h.shape[::-1])
import sys; sys.exit()
c,h = cv2.findContours(g.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print len(c)
c = sorted(c,key=cv2.contourArea,reverse=True)[:50]
for i,I in enumerate(c): print i,len(I)
cv2.drawContours(b,c,-1,(0,255,0),3)
b.astype(np.uint16).tofile('c_%dx%d.raw'%g.shape[::-1])

"""
http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html
https://simpleelastix.readthedocs.io/
g = np.gradient(b)
e = np.sqrt(g[0]*g[0]+g[1]*g[1]).astype(np.float32)
e.tofile('log/e_%dx%dx.raw'%e.shape[::-1])
"""