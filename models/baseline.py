'''The baseline model for household isolation of COVID-19'''
import numpy as np
from numpy import array, linspace
from numpy import sum as nsum
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from pickle import dump


class ModelSetup:
    def __init__(self):
        # From 2001 UK census the percentages of households from size 1 to 8 are:
        self.pages = array([
            30.28, 34.07, 15.51, 13.32, 4.88, 1.41, 0.33, 0.20])

# From 2001 UK census the percentages of households from size 1 to 8 are:
pages = array([
    30.28, 34.07, 15.51, 13.32, 4.88, 1.41, 0.33, 0.20])

weights = pages / nsum(pages)
nmax = len(weights)
nbar = weights @ np.arange(1, nmax + 1)

# INTERPRETABLE PARAMETERS:
latent_period = 5.0     # Days in E class before becoming infectious
prodrome_period = 3.0   # Days infectious but not symptomatic
infectious_period = 4.0 # Days infectious and symptomatic
RGp = 0.5               # Contribution to R0 outside the household from pre-symptomatic
RGi = 1.0               # Contribution to R0 outside the household during symptomatic
eta = 0.8               # Parameter of the Cauchemez model: HH transmission ~ n^(-eta)
import_rate = 0.001     # International importation rate
Npop = 5.6e7            # Total population
SAPp = 0.4              # Secondary attack probability for a two-person household with one susceptible and one prodrome
SAPi = 0.8              # Secondary attack probability for a two-person household with one susceptible and one infective
dist_start = 21         # Start of social distancing
dist_end = 42           # End of social distancing

s2i = np.zeros((nmax+1, nmax+1, nmax+1, nmax+1, nmax+1), dtype=np.int32)

k=0
for n in range(1, nmax+1):
    for s in range(0, n+1):
        for e in range(0, n+1-s):
            for p in range(0, n+1-s-e):
                for i in range(0, n+1-s-e-p):
                    s2i[n-1, s, e, p, i] = k
                    k += 1
imax = k

# Initialise the indices for sparse array generation

Ise = np.array([],dtype=np.int32)
Jse = np.array([],dtype=np.int32)
Vse = np.array([])

Ise_p = np.array([], dtype=np.int32)
Jse_p = np.array([], dtype=np.int32)
Vse_p = np.array([])

Ise_i = np.array([], dtype=np.int32)
Jse_i = np.array([], dtype=np.int32)
Vse_i = np.array([])

Iep = np.array([], dtype=np.int32)
Jep = np.array([], dtype=np.int32)
Vep = np.array([])

Ipi = np.array([],dtype=np.int32)
Jpi = np.array([], dtype=np.int32)
Vpi = np.array([])

Iir = np.array([], dtype=np.int32)
Jir = np.array([], dtype=np.int32)
Vir = np.array([])

ii = np.zeros(imax)
pp = np.zeros(imax)
ptilde = np.zeros(imax)

for n in range(1,nmax+1):
    for s in range(0, n+1):
        for e in range(0, n+1-s):
            for p in range(0, n+1-s-e):
                for i in range(0, n+1-s-e-p):
                    I = s2i[n-1, s, e, p, i]
                    
                    ii[I] = float(i)
                    pp[I] = float(p)
                    
                    if i==0:
                        ptilde[I] = float(p)
                    
                    if s > 0:
                        Ise = np.append(Ise,I)
                        Jse = np.append(Jse,s2i[n-1,s-1,e+1,p,i])
                        val = float(s)
                        Vse = np.append(Vse,val)
                        Ise = np.append(Ise,I)
                        Jse = np.append(Jse,I)
                        Vse = np.append(Vse,-val)   
                    if (s > 0) and (p > 0):
                        Ise_p = np.append(Ise_p,I)
                        Jse_p = np.append(Jse_p,s2i[n-1,s-1,e+1,p,i])
                        val = float(s*p)/(float(n)**(-eta)) # CAUCHEMEZ MODEL
                        Vse_p = np.append(Vse_p,val)
                        Ise_p = np.append(Ise_p,I)
                        Jse_p = np.append(Jse_p,I)
                        Vse_p = np.append(Vse_p,-val)      
                    if ((s>0) and (i>0)):
                        Ise_i = np.append(Ise_i,I)
                        Jse_i = np.append(Jse_i,s2i[n-1,s-1,e+1,p,i])
                        val = float(s*i)/(float(n)**(-eta)) # CAUCHEMEZ MODEL
                        Vse_i = np.append(Vse_i,val)
                        Ise_i = np.append(Ise_i,I)
                        Jse_i = np.append(Jse_i,I)
                        Vse_i = np.append(Vse_i,-val)                   
                    if (e>0):
                        Iep = np.append(Iep,I)
                        Jep = np.append(Jep,s2i[n-1,s,e-1,p+1,i])
                        val = float(e)
                        Vep = np.append(Vep,val)
                        Iep = np.append(Iep,I)
                        Jep = np.append(Jep,I)
                        Vep = np.append(Vep,-val)
                    if (p>0):
                        Ipi = np.append(Ipi,I)
                        Jpi = np.append(Jpi,s2i[n-1,s,e,p-1,i+1])
                        val = float(p)
                        Vpi = np.append(Vpi,val)
                        Ipi = np.append(Ipi,I)
                        Jpi = np.append(Jpi,I)
                        Vpi = np.append(Vpi,-val)
                    if (i>0):
                        Iir = np.append(Iir,I)
                        Jir = np.append(Jir,s2i[n-1,s,e,p,i-1])
                        val = float(i)
                        Vir = np.append(Vir,val)
                        Iir = np.append(Iir,I)
                        Jir = np.append(Jir,I)
                        Vir = np.append(Vir,-val)

Mse = csr_matrix((Vse, (Ise, Jse)), (imax, imax))
Mse_p = csr_matrix((Vse_p, (Ise_p, Jse_p)), (imax, imax))
Mse_i = csr_matrix((Vse_i, (Ise_i, Jse_i)), (imax, imax))
Mep = csr_matrix((Vep, (Iep, Jep)), (imax, imax))
Mpi = csr_matrix((Vpi, (Ipi, Jpi)), (imax, imax))
Mir = csr_matrix((Vir, (Iir, Jir)), (imax, imax))

rep = 1.0 / latent_period
rpi = 1.0 / prodrome_period
rir = 1.0 / infectious_period
beta_p = RGp / prodrome_period
beta_i = RGi / infectious_period
tau_p = (SAPp*rpi*(2.0**eta))/(1.0-SAPp)
tau_i = (SAPi*rir*(2.0**eta))/(1.0-SAPi)

# 0.04 is approximately an hour timestep for the implicit Euler numerical method
h = 0.04
Imat = eye(imax,imax)

# 0.04 is approximately an hour timestep for the implicit Euler numerical method
h = 0.04
Imat = eye(imax,imax)

trange = np.arange(0, 180, h)
tend = len(trange)

comply_range = linspace(0.0, 1.0, 6)
globred_range = array([0.0, 0.25, 0.5, 0.75])

lc = len(comply_range)
lg = len(globred_range)

prev = np.zeros((lg,lc,tend))
tint = 10.0 # Time at which we neglect imports
pdi = np.zeros((lg,lc,tend)) # Person-days in isolation
prav = np.zeros((nmax,lg,lc)) # Probability of avoiding by household size

for g in range(0, lg):
    for c in range(0, lc):
        irm = 1.0
        q0 = np.zeros(imax)
        for n in range(1, nmax+1):
            q0[s2i[n-1,n,0,0,0]] = weights[n-1]
        for t in range(0, tend):
            prev[g, c, t] = (ii @ q0)
            if t > 0:
                pdi[g, c, t] = \
                    pdi[g, c, t-1] + (comply_range[c] * prev[g,c,t]) * h
            rse = \
                irm * import_rate \
                + beta_p * (pp @ q0) \
                + (1.0 - comply_range[c]) * beta_i * (prev[g, c, t])
            if ((trange[t] >= dist_start) and (trange[t] <= dist_end)):
                rse = rse*(1.0 - globred_range[g])
            MM = \
                rse * Mse \
                + rep * Mep \
                + rpi * Mpi \
                + rir * Mir \
                + tau_p * Mse_p \
                + tau_i * Mse_i
            qh = spsolve(Imat - h * MM.T, q0)
            q0 = qh
            if (trange[t] >= tint):
                irm = 0.0

        for n in range(1,nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            prav[n-1,g,c] += s*q0[s2i[n-1,s,e,p,i]]
            prav[n-1,g,c] *= 1.0/(n*weights[n-1])
        
        print('Done global reduction range ' + str(g+1) + ' of ' + str(lg) + ' and compliance range ' + str(c+1) + ' of ' + str(lc))


with open('outputs.pkl', 'wb') as f:
    dump([
        lg, lc, trange, prev, nbar, Npop, comply_range,
        globred_range, tend, dist_start, dist_end, pdi], f)
