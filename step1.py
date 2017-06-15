import os,sys
data = sys.argv[1]
os.chdir('data/%s'%data)
(ix,iy) = eval(sys.argv[2])
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.misc import derivative
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,ConstantKernel,ExpSineSquared

sh = (Ns,Ny,Nx) = (100,1022,579)
flat = np.memmap('flat.raw',dtype=np.uint16,shape=sh,mode='r')

# Data
x = np.linspace(0,Ns,Ns)
xerr = 0.5*np.ones(Ns)
y = np.copy(flat[:,Ny/2+iy,Nx/2+ix])
yerr = np.sqrt(y)
  
# FFT
def g(x,A,w,phi,mval): return A*np.cos(w*x+phi)+mval
yF = np.fft.fft(y)
yF_norm = np.abs(yF)/Ns

# LE fit
xx = np.linspace(0,Ns,Ns*10)
"""
model = make_pipeline(PolynomialFeatures(5),RANSACRegressor(random_state=42))
model.fit(x[:,np.newaxis],y[:,np.newaxis])
"""

# GPR
alpha = 0.015
gpr_kernel = ExpSineSquared(length_scale=1.0,periodicity=100,periodicity_bounds=(90,110))#+ConstantKernel()+WhiteKernel(noise_level=0.001)
gpr = GaussianProcessRegressor(kernel=gpr_kernel,alpha=alpha)
#gpr.fit(np.atleast_2d(x).T,np.atleast_2d(y).T)
X = np.arange(3*Ns)
Y = np.tile(y,3)
gpr.fit(np.atleast_2d(X).T,np.atleast_2d(Y).T)
y_gpr = gpr.predict(np.atleast_2d(xx+Ns).T,return_std=False)

# Plot
fig,ax1 = plt.subplots()
ax1.errorbar(x,y,xerr=xerr,yerr=yerr,color='black',fmt='o',marker='o',markersize=4,linestyle='',label='data')
y0 = g(xx,2*yF_norm[1],2*np.pi/Ns,np.angle(yF)[1],yF_norm[0])
ax1.plot(xx,y0,'b',label='cosine')
y1 = y_gpr[:,0]
ax1.plot(xx,y1,'r',label='GPR')
ax1.set_xlabel('phase step')
ax1.set_ylabel('amplitude [ADC counts]')
plt.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(xx,y1-y0,color='orange',label='diff')
ax2.set_ylabel('diff')

plt.grid()
ax1.set_xlim(0,Ns)
plt.show()
raw_input()


#Difference
diff = y1-y0
fig,ax3 = plt.subplots()
difff = np.fft.fft(diff)#np.fft.fftshift(np.fft.fft(diff))

freq = np.fft.fftfreq(difff.shape[-1])
ax3.plot(freq, difff.real,'o')
#ax3.set_xlim(0,Ns)
plt.grid()
ax3.set_xlabel('phase step')
ax3.set_ylabel('Fourier coeff.')
plt.show()
raw_input()

#FFT Data


fig,ax4 = plt.subplots()
yf = np.fft.fft(y)#np.fft.fftshift(np.fft.fft(diff))

freq = np.fft.fftfreq(yf.shape[-1])
ax4.plot(freq, yf.real,'o')
#ax3.set_xlim(0,Ns)
plt.grid()
ax4.set_xlabel('phase step')
ax4.set_ylabel('Fourier coeff.')
plt.show()
raw_input()


