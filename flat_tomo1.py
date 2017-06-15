import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from pylab import *

#Data reading/Parameters

sh = (Nz,Ny,Nx) = (1464, 819, 846)
flat = np.memmap('flat_%dx%dx%d.raw'%sh[::-1], dtype=np.uint16, offset=0, shape=sh, mode='r')
N_ps = 8
mid = np.round([flat.shape[1] / 2 + 101, flat.shape[2] / 2 + 200])
print mid
#Functions definition

def g(_x, _A, _w, _phi, _mval): return _A * np.cos(_w * _x + _phi) + _mval

def get_cosCurve(g_psF, x):
	g_psF_norm = np.abs(g_psF) / N_ps
	A = 2 * g_psF_norm[1]
	w = 2 * np.pi / N_ps
	phi = np.angle(g_psF)[1]
	mval = g_psF_norm[0]
	g_a = g(x, A, w, phi, mval)
	return g_a, phi, A, mval

#Variable initiallization
	
Nf = Nz / N_ps
x = np.linspace(0, N_ps, 1000)
g_p = np.zeros(N_ps)
phis = np.zeros(Nf, np.float32)
As = np.zeros(Nf, np.float32)
mvals = np.zeros(Nf, np.float32)
phia = np.zeros((Nf, Ny, Nx), np.float32)
mvala = np.zeros((Nf, Ny, Nx), np.float32)
Aa = np.zeros((Nf, Ny, Nx), np.float32)

#Looping over all projections, processing and writing files

for i in range(Nf):	
    fflat = np.fft.fft(flat[i * N_ps:(i + 1) * N_ps], axis=0)
    g_p = fflat[:, mid[0], mid[1]]
    (g_a, phi, A, mval) = get_cosCurve(g_p, x)
    phis[i] = phi
    As[i] = A
    mvals[i] = mval
    phia[i] = np.angle(fflat[1])
    Aa[i] = abs(fflat[1])
    mvala[i] = abs(fflat[0])
    plt.plot(x, g_a)
    del fflat

mvala.astype(np.float32).tofile('log_m/m_val_%dx%dx%d.raw'%mvala.shape[::-1])
Aa.astype(np.float32).tofile('log_A/A_%dx%dx%d.raw'%Aa.shape[::-1])
phia.astype(np.float32).tofile('log_phi/phi_%dx%dx%d.raw'%phia.shape[::-1])

phia = np.diff(phia, axis=0)
phix = np.mean(phia, axis=(1, 2))
phia = (np.mod(phia + np.pi, 2 * np.pi)) - np.pi
phiam = np.minimum(phia, phia + 2 * np.pi)

mvala = (mvala - mvala[0]) / np.mean(mvala, axis=0)
Aa = (Aa - Aa[0]) / np.mean(Aa, axis=0)

mvala.astype(np.float32).tofile('log_m/mval_norm_%dx%dx%d.raw'%mvala.shape[::-1])
Aa.astype(np.float32).tofile('log_A/A_norm_%dx%dx%d.raw'%Aa.shape[::-1])
phia.astype(np.float32).tofile('log_phi/phi_norm_%dx%dx%d.raw'%phia.shape[::-1])
phiam.astype(np.float32).tofile('log_phi/phi_min_%dx%dx%d.raw'%phiam.shape[::-1])

phia = np.mean(phia, axis=(1, 2))
phia = np.cumsum(phia)
mvala = np.mean(mvala, axis=(1, 2))
Aa = np.mean(Aa, axis=(1, 2))

phis = np.diff(phis, axis=0)
phis = (np.mod(phis + np.pi, 2 * np.pi)) - np.pi
phis = np.cumsum(phis)

#Plotting

plt.title('Phase stepping for a single pixel (FFT)')
plt.xlabel('Phase step')
ylabel('Pix. value')
plt.grid()
plt.show()
plt.savefig('pscurve2.png')
raw_input()

plt.subplot(3, 1, 1)
plt.plot(phis, 'bo', label=r'$\Delta \phi$ (one pixel)')
plt.xlabel('Projection')
plt.ylabel(r'$\Delta \phi$')
plt.grid()
plt.xlim(0, Nf)
plt.legend(prop={'size':10})	
	
plt.subplot(3, 1, 2)
plt.plot((mvals - mvals[0]) / mvals.mean(), label=r'$\Delta$ DC (one pixel)')
plt.xlabel('Projection')
plt.ylabel(r'$\Delta$ DC')
plt.grid()
plt.xlim(0, Nf)
plt.legend(prop={'size':10})	
	
plt.subplot(3, 1, 3)
plt.plot((As - As[0]) / As.mean(), label=r'$\Delta$ Amplitude (one pixel)')
plt.xlabel('Projection')
plt.ylabel(r'$\Delta$ Amplitude')
plt.grid()
plt.xlim(0, Nf)
plt.legend(prop={'size':10})	

plt.subplot(3, 1, 1)
plt.plot(phia, 'go', label=r'$\Delta \phi$ (all pixels)')
plt.xlabel('Projection')
plt.ylabel(r'$\Delta \phi$')
plt.xlim(1, Nf)
plt.subplot(3, 1, 1)
plt.plot(phix, 'ko', label=r'$\Delta \phi$ (all pixels not cumm)')
plt.xlabel('Projection')
plt.ylabel(r'$\Delta \phi$')
plt.xlim(1, Nf)
plt.legend(loc=4, prop={'size':10})	
	
plt.subplot(3, 1, 2)
plt.plot(mvala, label=r'$\Delta$ DC (all pixels)')
plt.xlabel('Projection')
plt.ylabel(r'$\Delta$ DC')
plt.xlim(0, Nf)
plt.legend(prop={'size':10})	
	
plt.subplot(3, 1, 3)
plt.plot(Aa, label=r'$\Delta$ Amplitude (all pixels)')
plt.xlabel('Projection')
plt.ylabel(r'$\Delta$ Amplitude')
plt.xlim(0, Nf)
plt.legend(prop={'size':10})	

plt.show()
plt.savefig('differences2.png')
raw_input()
