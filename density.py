import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

d_x = 0.0041
data_name = sys.argv[1]
sh = (Nz, Ny, Nx) = (int(sys.argv[2]), 500, 500)
measdata = np.memmap('Bones/' + data_name + '_%dx%dx%d.raw'%sh[::-1], dtype=np.float32, offset=0, shape=sh, mode='r').astype(np.float32)
mval = np.mean(measdata, axis=(1, 2))

#Force(skip header=2 normally, =3 for sample 3)
csv = np.genfromtxt('force/CaseNr %s/KalotteNo%s.csv'%(sys.argv[3], sys.argv[3]), delimiter=';', skip_header=2)
st_weg = csv[:, -1]
st_force = csv[:, -2]

#rho = 4245.5 * mval - 791.1 #for first measurement
rho = 4226.9887 * mval - 760.0887#for second measurement
rho = rho[np.where(rho > 0)]
x_vec = np.arange(0, len(rho)).astype(np.float32)
x_vec *= d_x
x_vec2 = np.max(x_vec) - (st_weg / 10)
x_vec = x_vec[np.where(x_vec > np.min(x_vec2))]
m, b, r_val, p_val, std_err = stats.linregress(st_weg, st_force)
rho = rho[-1:-len(x_vec) - 1:-1]
rho = rho[::-1]
integral = np.trapz(rho)
norm = x_vec[-1]
norm_int = integral / norm

"""
First approach
#Density
#rho = 4245.5 * mval - 791.1 #for first measurement
rho = 4226.9887 * mval - 760.0887#for second measurement
rho = rho[np.where(rho > 0)]
x_vec = np.arange(0, len(rho)).astype(np.float32)
x_vec *= d_x
x_vec2 = np.max(x_vec) - (st_weg / 10)
integral = np.trapz(rho)
norm = x_vec[-1]
norm_int = integral / norm
"""

fig, ax1 = plt.subplots()
ax1.plot(x_vec, rho, label='Integral=' + str(norm_int) +  ' (norm)' + '\n               ' + str(integral), color='b')
plt.legend(loc=2)
ax1.set_xlabel('Depth(cm)')
ax1.set_ylabel(r'Density (mg HA/ cm$^{3}$)', color='b')
ax1.set_ylim([0, 900])
ax2 = ax1.twinx()
ax2.plot(x_vec2, st_force, label='slope=' + str(m), color='g')
ax2.set_ylabel('Force (N)', color='g')
plt.grid()
plt.legend(loc=3)
ax2.set_xlim([0, 2])
ax2.set_ylim([0, 1200])
plt.title(data_name)
plt.savefig(data_name + '.png')

