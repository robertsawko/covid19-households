'''The baseline model for household isolation of COVID-19

Notation of the essential variables follows the manuscript to improve the readability.
'''
from abc import ABC
from numpy import append, arange, array, int32, zeros
from numpy import sum as nsum
from numpy import max as nmax
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve


class BasicModelSetup:
    '''This class holds all the fixed parameters of the model'''
    def __init__(self):
        # From 2001 UK census the percentages of households from size 1 to 8 are:
        pages = array([
            30.28, 34.07, 15.51, 13.32, 4.88, 1.41, 0.33, 0.20])

        self.weights = pages / nsum(pages)
        self.nmax = len(self.weights)
        self.nbar = self.weights @ arange(1, self.nmax + 1)

        # INTERPRETABLE PARAMETERS:
        latent_period = 5.0     # Days in E class before becoming infectious
        prodrome_period = 3.0   # Days infectious but not symptomatic
        infectious_period = 4.0 # Days infectious and symptomatic
        RGp = 0.5               # Contribution to R0 outside the household from pre-symptomatic
        RGi = 1.0                   # Contribution to R0 outside the household during symptomatic
        eta = 0.8                   # Parameter of the Cauchemez model: HH transmission ~ n^(-eta)
        self.import_rate = 0.001    # International importation rate
        self.Npop = 5.6e7       # Total population
        SAPp = 0.4              # Secondary attack probability for a two-person household with one susceptible and one prodrome
        SAPi = 0.8              # Secondary attack probability for a two-person household with one susceptible and one infective
        self.dist_start = 21         # Start of social distancing
        self.dist_end = 42           # End of social distancing

        self.s2i = zeros((
            self.nmax+1, self.nmax+1, self.nmax+1, self.nmax+1, self.nmax+1),
            dtype=int32)

        k=0
        for n in range(1, self.nmax+1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            self.s2i[n-1, s, e, p, i] = k
                            k += 1
        self.imax = k

        # Initialise the indices for sparse array generation
        Ise = array([], dtype=int32)
        Jse = array([], dtype=int32)
        Vse = array([])

        Ise_p = array([], dtype=int32)
        Jse_p = array([], dtype=int32)
        Vse_p = array([])

        Ise_i = array([], dtype=int32)
        Jse_i = array([], dtype=int32)
        Vse_i = array([])

        Iep = array([], dtype=int32)
        Jep = array([], dtype=int32)
        Vep = array([])

        Ipi = array([],dtype=int32)
        Jpi = array([], dtype=int32)
        Vpi = array([])

        Iir = array([], dtype=int32)
        Jir = array([], dtype=int32)
        Vir = array([])

        self.ii = zeros(self.imax)
        self.pp = zeros(self.imax)
        self.ptilde = zeros(self.imax)

        for n in range(1, self.nmax + 1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            I = self.s2i[n-1, s, e, p, i]
                            self.ii[I] = float(i)
                            self.pp[I] = float(p)
                            
                            if i==0:
                                self.ptilde[I] = float(p)
                            
                            if s > 0:
                                Ise = append(Ise, I)
                                Jse = append(Jse, self.s2i[n-1, s-1, e+1, p, i])
                                val = float(s)
                                Vse = append(Vse, val)
                                Ise = append(Ise, I)
                                Jse = append(Jse, I)
                                Vse = append(Vse, -val)   
                            if (s > 0) and (p > 0):
                                Ise_p = append(Ise_p, I)
                                Jse_p = append(Jse_p, self.s2i[n-1,s-1,e+1,p,i])
                                val = float(s*p)/(float(n)**(-eta)) # CAUCHEMEZ MODEL
                                Vse_p = append(Vse_p, val)
                                Ise_p = append(Ise_p, I)
                                Jse_p = append(Jse_p, I)
                                Vse_p = append(Vse_p, -val)      
                            if (s > 0) and (i > 0):
                                Ise_i = append(Ise_i, I)
                                Jse_i = append(Jse_i, self.s2i[n-1,s-1,e+1,p,i])
                                val = float(s*i)/(float(n)**(-eta)) # CAUCHEMEZ MODEL
                                Vse_i = append(Vse_i, val)
                                Ise_i = append(Ise_i, I)
                                Jse_i = append(Jse_i, I)
                                Vse_i = append(Vse_i,-val)                   
                            if e > 0:
                                Iep = append(Iep, I)
                                Jep = append(Jep, self.s2i[n-1,s,e-1,p+1,i])
                                val = float(e)
                                Vep = append(Vep,val)
                                Iep = append(Iep,I)
                                Jep = append(Jep,I)
                                Vep = append(Vep,-val)
                            if p > 0:
                                Ipi = append(Ipi, I)
                                Jpi = append(Jpi, self.s2i[n-1,s,e,p-1,i+1])
                                val = float(p)
                                Vpi = append(Vpi,val)
                                Ipi = append(Ipi,I)
                                Jpi = append(Jpi,I)
                                Vpi = append(Vpi,-val)
                            if i > 0:
                                Iir = append(Iir, I)
                                Jir = append(Jir, self.s2i[n-1,s,e,p,i-1])
                                val = float(i)
                                Vir = append(Vir,val)
                                Iir = append(Iir,I)
                                Jir = append(Jir,I)
                                Vir = append(Vir,-val)

        self.matrix_size = (self.imax, self.imax)
        self.Mse = csr_matrix((Vse, (Ise, Jse)), self.matrix_size)
        self.Mse_p = csr_matrix((Vse_p, (Ise_p, Jse_p)), self.matrix_size)
        self.Mse_i = csr_matrix((Vse_i, (Ise_i, Jse_i)), self.matrix_size)
        self.Mep = csr_matrix((Vep, (Iep, Jep)), self.matrix_size)
        self.Mpi = csr_matrix((Vpi, (Ipi, Jpi)), self.matrix_size)
        self.Mir = csr_matrix((Vir, (Iir, Jir)), self.matrix_size)

        self.rep = 1.0 / latent_period
        self.rpi = 1.0 / prodrome_period
        self.rir = 1.0 / infectious_period
        self.beta_p = RGp / prodrome_period
        self.beta_i = RGi / infectious_period
        self.tau_p = (SAPp * self.rpi * (2.0**eta)) / (1.0 - SAPp)
        self.tau_i = (SAPi * self.rir * (2.0**eta)) / (1.0 - SAPi)

        # 0.04 is approximately an hour timestep for the implicit Euler numerical method
        self.h = 0.04
        self.Imat = eye(*self.matrix_size)

        self.trange = arange(0, 180, self.h)
        self.tint = 10.0                     # Time at which we neglect imports


class HouseholdModel(ABC):
    def __init__(self):
        pass

    def plot_cases(self, axes, label, colour):
        axes.plot(
            self.setup.trange,
            (self.setup.Npop / self.setup.nbar) * self.prev,
            label=label,
            c=colour)

    def peak_ratio(self, other):
        return nmax(self.prev)/nmax(other.prev)

    @property
    def peak_value(self):
        return (self.setup.Npop / self.setup.nbar) * nmax(self.prev)

    @property
    def persdays(self):
        return (self.setup.Npop / self.setup.nbar) * self.pdi[-2]

    @property
    def max_person_days_of_isolation(self):
        return (self.setup.Npop / self.setup.nbar) * nmax(self.pdi)

    def plot_person_days_of_isolation(self, axes, label, colour):
        axes.plot(
            self.setup.trange,
            (self.setup.Npop / self.setup.nbar) * self.pdi,
            label=label,
            c=colour)



class IndividualIsolationModel(HouseholdModel):
    '''Individual isolation assumes that sympotomatic cases self-isolate and
    cease transmission outside of the household.
    '''
    def __init__(self, setup, epsilon, alpha_c):
        self.compliance = alpha_c
        self.global_reduction = epsilon
        self.setup = setup

        self.prev = zeros(len(setup.trange))
        self.pdi = zeros(len(setup.trange))     # Person-days in isolation
        prav = zeros(setup.nmax)    # Probability of avoiding by household size

        irm = 1.0
        q0 = zeros(setup.imax)
        for n in range(1, setup.nmax + 1):
            q0[setup.s2i[n-1, n, 0, 0, 0]] = setup.weights[n - 1]
        for ti, t in enumerate(setup.trange):
            self.prev[ti] = (setup.ii @ q0)
            if ti > 0:
                self.pdi[ti] = \
                    self.pdi[ti-1] + (alpha_c * self.prev[ti]) * setup.h
            rse = \
                irm * setup.import_rate \
                + setup.beta_p * (setup.pp @ q0) \
                + (1.0 - alpha_c) * setup.beta_i * self.prev[ti]

            if setup.dist_start <= t <= setup.dist_end:
                rse = rse * (1.0 - epsilon)

            MM = \
                rse * setup.Mse \
                + setup.rep * setup.Mep \
                + setup.rpi * setup.Mpi \
                + setup.rir * setup.Mir \
                + setup.tau_p * setup.Mse_p \
                + setup.tau_i * setup.Mse_i
            qh = spsolve(setup.Imat - setup.h * MM.T, q0)
            q0 = qh
            if (t >= setup.tint):
                irm = 0.0

        for n in range(1, setup.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            prav[n-1] += s * q0[setup.s2i[n-1, s, e, p, i]]
            prav[n - 1] *= 1.0 /(n * setup.weights[n-1])
        
class WeakHouseholdIsolationModel(HouseholdModel):
    '''Weak household isolation model assumes a compliant percentage of
    househods will isolate if there is at least one symptomatic case'''
    def __init__(self, setup, epsilon, alpha_c):
        self.compliance = alpha_c
        self.global_reduction = epsilon
        self.setup = setup

        self.prev = zeros(len(setup.trange))
        self.pdi = zeros(len(setup.trange))     # Person-days in isolation
        prav = zeros(setup.nmax)    # Probability of avoiding by household size
        # Overwrite the susceptible to exposed matrix
        Ise = array([],dtype=int32)
        Jse = array([],dtype=int32)
        Vse = array([])
        for n in range(1, setup.nmax + 1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            I = setup.s2i[n-1,s,e,p,i]
                            if (s>0):
                                Ise = append(Ise, I)
                                Jse = append(
                                    Jse, setup.s2i[n-1,s-1,e+1,p,i])
                                if (i==0):
                                    val = float(s)
                                else:
                                    val = (1.0 - alpha_c) * float(s)
                                Vse = append(Vse, val)
                                Ise = append(Ise, I)
                                Jse = append(Jse, I)
                                Vse = append(Vse, -val)  
        setup.Mse = csr_matrix((Vse, (Ise, Jse)), setup.matrix_size)
            
        irm = 1.0
        q0 = zeros(setup.imax)
        for n in range(1,setup.nmax+1):
            q0[setup.s2i[n-1,n,0,0,0]] = setup.weights[n-1]
        for ti, t in enumerate(setup.trange):
            self.prev[ti] = (setup.ii @ q0)
            if (t>0):
                self.pdi[ti] = \
                    self.pdi[ti-1] + (alpha_c * self.prev[ti]) * setup.h
                
            rse = \
                irm * setup.import_rate \
                + (1.0 - alpha_c) * (
                    setup.beta_p * (setup.pp @ q0)
                    + setup.beta_i*(self.prev[ti])) \
                + alpha_c * setup.beta_p * (setup.ptilde @ q0)
        
            if setup.dist_start <= t <= setup.dist_end:
                rse = rse*(1.0 - epsilon)
            
            MM = \
                rse * setup.Mse \
                + setup.rep * setup.Mep \
                + setup.rpi * setup.Mpi \
                + setup.rir * setup.Mir \
                + setup.tau_p * setup.Mse_p \
                + setup.tau_i * setup.Mse_i
            qh = spsolve(setup.Imat - setup.h * MM.T, q0)
            q0 = qh
            if (t >= setup.tint):
                irm = 0.0

        for n in range(1, setup.nmax+1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            prav[n-1] += s * q0[setup.s2i[n-1, s, e, p, i]]
            prav[n-1] *= 1.0 / (n * setup.weights[n-1])
