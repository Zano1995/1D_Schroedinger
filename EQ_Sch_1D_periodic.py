import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.constants as cs
import scipy 
from scipy import signal
from cmath import phase
import matplotlib.colors as mcolors
from time import time
def mat_diag(k_list,g_max,V_G_G,M,a):
	g_dim = 2*np.pi/a
	num_k_list = np.shape(k_list)[0]
	matr = np.zeros((2*g_max+1,2*g_max+1, num_k_list), dtype="complex")
	for k_ind, k in enumerate(k_list):
		for i in range(np.shape(matr)[0]):
			for j in range(np.shape(matr)[1]):
				if i==j:#########      Diagonal terms  [(k+nG) ..... (k) ..... (k-nG)]
					matr[i,j,k_ind] = (cs.hbar)**2*(k+(g_max-i)*g_dim)**2/(2*M*cs.e)
				else:          ##### off diagonal terms
					if j>i:
						matr[i,j,:] = V_G_G[(j-i)]
	return matr

def get_bands(mat):
	mat_dim = np.shape(mat)[0]       # matrix dimension, it corresponds to 2*g_max+1
	num_k = np.shape(mat)[2]         # number of k points calculated
	bands = np.zeros((num_k,mat_dim))
	losses = np.zeros((num_k,mat_dim ))
	coeff_wavef = np.zeros((num_k,mat_dim,mat_dim ), dtype="complex")
	for k in range(num_k):
		eigv, eigvec = la.eigh(mat[:,:,k],UPLO='U')
		bands[k,:] = np.real(eigv)
		losses[k,:] = np.imag(eigv)
		coeff_wavef[k,:,:] = eigvec	
	return bands, losses, coeff_wavef
def wavef(k_arr,k_ind,n_ind,x,coeff,period):
	len_x = np.shape(x)[0]
	len_k = np.shape(coeff)[1]
	g_maximum = (len_k-1)//2
	func = np.zeros((len_x), dtype="complex")
	for i, c_k in enumerate(coeff[k_ind,:,n_ind]):
		g_ind = (g_maximum-i)
		func += c_k*np.exp(1j*(k_arr[k_ind]+g_ind*2*np.pi/period)*x)
	return func

def inverse(dim_real, Pot_coeff): ## We manually construct the inverse transform
	itf = np.zeros(dim_real)
	for n,coeff_pot in enumerate(V_coeff):
		itf += np.real(coeff_pot)*np.cos(2*np.pi*n/a*x)-np.imag(coeff_pot)*np.sin(2*np.pi*n/a*x)
	return itf

def Fourier_pot(V,g_max):
	dim = np.shape(V)[0]
	ft = scipy.fft.rfft(V)/dim*2
	V_coeff = ft[0:2*g_max+1]   
	V_coeff[0]=V_coeff[0]/2   
	return V_coeff 

def square(x_arr, fill_f,off, V_maximum):
	dim_x = np.shape(x_arr)[0]
	V = np.zeros(dim_x)
	V[int(dim_x*offset):int(dim_x*(offset+fill_f))] = V_maximum
	return V
def triangular(x_arr,a, V_maximum):
	dim_x = np.shape(x_arr)[0]
	V = V_maximum*(a/2-np.abs(x-a/2))*2/a
	return V


num_k = 201     #number of k points in BZ
g_max = 20    # max reciporcal lattice vector in units of 2*pi/a
a= 5*10**(-10)  #lattice constant
M = cs.m_e      #mass
dim = 4096*8      # division in real space of the potential
x=np.linspace(0,a,dim)  # x points of unit cell in real space 
V_max = 0.8
fill_fraction = 0.8
offset = 0.4

V = square(x,fill_fraction,offset,V_max)

 
V_coeff = Fourier_pot(V,g_max)
itf = inverse(dim, V_coeff)
k=np.linspace(-np.pi/a,np.pi/a,num_k)


mat_prova = mat_diag(k,g_max,V_G_G=V_coeff, M=M, a=a)
bands, losses, coeff_wave_arr = get_bands(mat_prova)

######   wavefunction
n = 1
k_index = 200
func = wavef(k,k_index,n,x,coeff_wave_arr,a)
prob_dens = np.abs(func)**2
norm_dens = np.max(prob_dens)


########### plot
plt.title(f"Potential and wavefunction in $k a/\\pi=${k[k_index]*a/np.pi:.2f}, n= {n+1} ")
plt.plot(x/a, prob_dens,"--", label="$|\\psi|^2$", color="blue")
plt.plot(x/a+1, prob_dens,"--", color="blue")
plt.plot(x/a,V,label="Potential", color="red")
plt.plot(x/a+1,V, color="red")
plt.plot(x/a,itf,label="Inverse FT", color="green")
plt.plot(x/a+1,itf, color="green")
plt.grid(alpha=0.5)
plt.legend()
plt.xlabel("$x/a$")
plt.ylabel("Potential (eV)")
plt.show()

plt.xlabel("Wavevector $k_x a/\\pi$")
plt.ylabel("Energy (eV)")
plt.plot(k*a/np.pi,bands)
plt.ylim((-1,10))
plt.grid(alpha=0.5)
plt.show()