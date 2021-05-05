import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import matplotlib.pyplot as plt
import time 

#Constants
hbar = 1.0
m_e = 0.511e6
e = 0.303
eps_0 = 1.0
a_0 = 2.7e-4
Z = 1
E0 = 13.6

N = int(1e3)
dr = 6 * a_0 / (N-1)
dtheta = 0.01
n = 0

R = 2.1 * a_0

#----------------------

#Functions

def Overlap(arr_1, arr_2):
    integral = 0
    for radius in range(0, len(arr_1)):
        for angle in frange(0, np.pi, dtheta):
            #Units of Bohr Radii
            r_a = radius * dr
            r_b = np.sqrt(r_a**2 + R**2 - (2 * r_a * R * np.cos(angle)))
            #Switch to units of stepsize
            if round(r_b/dr) > len(arr_1):
                integral += 0
            elif round(r_b/dr) < len(arr_1):
                integral += arr_1[round(r_a/dr)] * arr_2[round(r_b/dr)] * (radius * dr)**2 * np.sin(angle)
    integral = integral * 2*np.pi * dr * dtheta
    return integral

def Norm_single(array):
    J_array = np.zeros(len(array))
    for i in range(len(array)):
        J_array[i] = ((dr*i)**2) * array[i]
    A_single = 1.0 / np.sqrt(4*np.pi * dr * np.dot(J_array, array))
    norm_array = A_single * array
    return norm_array

def frange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def Kinetic(array):
    array = diags([-2, 1, 1], [0, -1, 1], shape = [N,N]).toarray()
    array = (-hbar**2)/(2 * m_e * dr**2) * array
    return array
    
def Potential_1(array):
    r_0 = 1e-6
    for i in range(1, N):
        array[i, i] = -1.0 / (i * dr)
    #array[0,0] = ((-1.0/r_0) + (1.0/r_0**2) * (0 - r_0) - (1/2)*(1/r_0**3)*(0 - r_0)**2)
    array = (Z*e**2 / (4*np.pi*eps_0)) * array
    return array
    
def Potential_2(array):
    for i in range(0, N):
        integral = 0
        r_1 = i * dr
        for alpha in frange(0, np.pi, dtheta):
            r_2 = np.sqrt(r_1**2 + R**2 - 2*r_1*R*np.cos(alpha))
            integral += (-1.0 / r_2) * np.sin(alpha)
        array[i, i] = integral * dtheta * (1/2)
        #print(i, array[i,i])
    array = (Z*e**2 / (4*np.pi*eps_0)) * array
    return array
    
def Energy_pert(arr_1, arr_2):
    D = X = 0
    #Direct, Exchange Integrals
    for radius in range(0, len(arr_1)):
        for angle in frange(0, np.pi, dtheta):
            r_a = radius * dr
            r_b = np.sqrt(r_a**2 + R**2 - (2 * r_a * R * np.cos(angle)))
            if round(r_b/dr) > len(arr_1):
                D += 0
                X += 0
            elif round(r_b/dr) < len(arr_1):
                D += arr_1[radius]**2 * (radius * dr)**2 * np.sin(angle) * (1.0 / (r_b))
                X += arr_1[round(r_a/dr)] * arr_2[round(r_b/dr)] * (radius * dr) * np.sin(angle)
    D = D * 2*np.pi * dr * dtheta * a_0
    X = X * 2*np.pi * dr * dtheta * a_0
    return D, X
        
#----------------------


init_array = np.zeros([N,N], dtype = float)

#Hamiltonian
H = Kinetic(init_array) + Potential_1(init_array) + Potential_2(init_array)


#<V> = -18264.82021290479
#<T> = 9.637853890996684

#Eigenvalues
eigvals, eigvecs = eigsh(H, k = 1, which = 'SA')

print("R/a_0:", R/a_0, "Numeric Energy:",  eigvals[0] + (e**2 / (4*np.pi * eps_0 * R)))

x = R/a_0
F = -1 + (2/x) * (( (1 - (2/3)*x**2)*np.exp(-x) + (1 + x)*np.exp(-2*x)) / (1 + (1 + x + (1/3)*x**2)*np.exp(-x)))
print("R/a_0:", R/a_0, "Analytic Energy:", F*E0)

r1 = np.linspace(0, N, N)
r2 = np.linspace(0, N, N)

psi = np.ones([2, N])
psi[0] = Norm_single(np.exp(-r1*dr/a_0))
psi[1] = Norm_single(np.exp(-r2*dr/a_0))

e_pert = Energy_pert(psi[0], psi[1])[0] + Energy_pert(psi[0], psi[1])[1]
print("pert:", eigvals[0] + (e**2 / (4*np.pi * eps_0 * R)) - e_pert)

'''rad = []
En = []
F_list = []

for k in range(12):
    H = Kinetic(init_array) + Potential_1(init_array) + Potential_2(init_array)
    eigvals, eigvecs = eigsh(H, k = 1, which = 'SA')
    
    rad.append(R/a_0)
    En.append(eigvals[0] + (e**2 / (4*np.pi * eps_0 * R)))
    
    x = R/a_0
    F = -1 + (2/x) * (( (1 - (2/3)*x**2)*np.exp(-x) + (1 + x)*np.exp(-2*x)) / (1 + (1 + x + (1/3)*x**2)*np.exp(-x)))
    F_list.append(F*E0)
    
    print(R/a_0, En[k], F_list[k])
    R += 0.5 * a_0
    
    
    
    
plt.plot(rad, En, 'o', color = 'black', label = 'Numeric Energy')
plt.plot(rad, F_list, 'o', color = 'red', label = 'Analytic Energy')
plt.xlabel('$R/a_0$')
plt.ylabel('$<H>_{min}$')
plt.legend(loc = 'upper right')
plt.show()'''
  


