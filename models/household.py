'''The household isolation models for COVID-19 epidemic in the UK. All models
assume:
    * non-pharmaceutical interventions (NPI) of length T.

Notation of the essential variables follows the manuscript to improve the
readability.

Note: some repetition is allowed for readability e.g. epsilon and alpha_c
assignment are repeated in every constructor.
'''
from abc import ABC, abstractmethod
from copy import deepcopy
from numpy import append, arange, array, int32, zeros
from numpy import sum as nsum
from numpy import max as nmax
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from models.configs import DEFAULT_PARAMS

class Setup(ABC):
    def __init__(self, params):
        self.params = deepcopy(params)
        self.trange = arange(
            0, params['final_time'], params['h'])

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def NPI_active(self, t):
        '''Check if non-pharmaceutical interventions are active at time t'''
        return self.params['npi']['start'] <= t <= self.params['npi']['end']


class BasicModelSetup(Setup):
    '''This class produces matrix for a 5 compartment models'''
    def __init__(self, params=DEFAULT_PARAMS):
        super().__init__(params)
        self.weights = params['pages'] / nsum(params['pages'])
        self.nmax = len(self.weights)
        self.nbar = self.weights @ arange(1, self.nmax + 1)

        # A mapping of multi-index onto a linear index
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
                                val = float(s*p)/(float(n)**(-params['eta'])) # CAUCHEMEZ MODEL
                                Vse_p = append(Vse_p, val)
                                Ise_p = append(Ise_p, I)
                                Jse_p = append(Jse_p, I)
                                Vse_p = append(Vse_p, -val)      
                            if (s > 0) and (i > 0):
                                Ise_i = append(Ise_i, I)
                                Jse_i = append(Jse_i, self.s2i[n-1,s-1,e+1,p,i])
                                val = float(s*i)/(float(n)**(-params['eta'])) # CAUCHEMEZ MODEL
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
                            if i > 0:
                                Iir = append(Iir, I)
                                Jir = append(Jir, self.s2i[n-1,s,e,p,i-1])
                                val = float(i)
                                Vir = append(Vir,val)
                                Iir = append(Iir,I)
                                Jir = append(Jir,I)
                                Vir = append(Vir,-val)
                            if p > 0:
                                Ipi = append(Ipi, I)
                                Jpi = append(Jpi, self.s2i[n-1,s,e,p-1,i+1])
                                val = float(p)
                                Vpi = append(Vpi,val)
                                Ipi = append(Ipi,I)
                                Jpi = append(Jpi,I)
                                Vpi = append(Vpi,-val)

        self.matrix_size = (self.imax, self.imax)
        self.Mse = csr_matrix((Vse, (Ise, Jse)), self.matrix_size)
        self.Mse_p = csr_matrix((Vse_p, (Ise_p, Jse_p)), self.matrix_size)
        self.Mse_i = csr_matrix((Vse_i, (Ise_i, Jse_i)), self.matrix_size)
        self.Mep = csr_matrix((Vep, (Iep, Jep)), self.matrix_size)
        self.Mpi = csr_matrix((Vpi, (Ipi, Jpi)), self.matrix_size)
        self.Mir = csr_matrix((Vir, (Iir, Jir)), self.matrix_size)

        self.rep = 1.0 / params['latent_period']
        self.rpi = 1.0 / params['prodrome_period']
        self.rir = 1.0 / params['infectious_period']
        self.beta_p = params['RGp'] / params['prodrome_period']
        self.beta_i = params['RGi'] / params['infectious_period']
        self.tau_p = (
            params['SAPp'] * self.rpi * (2.0**self.params['eta'])) \
            / (1.0 - params['SAPp'])
        self.tau_i = (
            params['SAPi'] * self.rir * (2.0**params['eta'])) \
            / (1.0 - self.params['SAPi'])

        self.Imat = eye(*self.matrix_size)

class HouseholdModel(ABC):
    def __init__(self, setup):
        self.global_reduction = setup['npi']['global_reduction']
        self.setup = deepcopy(setup)

        self.prev = zeros(len(setup.trange))
        self.pdi = zeros(len(setup.trange))     # Person-days in isolation
        self.prav = zeros(setup.nmax)           # Probability of avoiding by household size


    def plot_cases(self, axes, label, colour):
        axes.plot(
            self.setup.trange,
            (self.setup['Npop'] / self.setup.nbar) * self.prev,
            label=label,
            c=colour)

    def peak_ratio(self, other):
        return nmax(self.prev)/nmax(other.prev)

    @property
    def peak_value(self):
        return (self.setup['Npop'] / self.setup.nbar) * nmax(self.prev)

    @property
    def persdays(self):
        return (self.setup['Npop'] / self.setup.nbar) * self.pdi[-2]

    @property
    def max_person_days_of_isolation(self):
        return (self.setup['Npop'] / self.setup.nbar) * nmax(self.pdi)

    def plot_person_days_of_isolation(self, axes, label, colour):
        axes.plot(
            self.setup.trange,
            (self.setup['Npop'] / self.setup.nbar) * self.pdi,
            label=label,
            c=colour)

    @abstractmethod
    def solve(self):
        pass

class IndividualIsolationModel(HouseholdModel):
    '''Individual isolation assumes that sympotomatic cases self-isolate and
    cease transmission outside of the household.
    '''
    def __init__(self, setup):
        super().__init__(setup)

    def solve(self):
        alpha_c = self.setup['npi']['compliance']
        epsilon = self.setup['npi']['global_reduction']

        irm = 1.0
        q0 = zeros(self.setup.imax)
        for n in range(1, self.setup.nmax + 1):
            q0[self.setup.s2i[n-1, n, 0, 0, 0]] = self.setup.weights[n - 1]
        for ti, t in enumerate(self.setup.trange):
            self.prev[ti] = (self.setup.ii @ q0)
            if ti > 0:
                self.pdi[ti] = \
                    self.pdi[ti-1] + (alpha_c * self.prev[ti]) * self.setup['h']
            rse = \
                irm * self.setup['import_rate'] \
                + self.setup.beta_p * (self.setup.pp @ q0) \
                + (1.0 - alpha_c) * self.setup.beta_i * self.prev[ti]

            if self.setup.NPI_active(t):
                rse = rse * (1.0 - epsilon)

            MM = \
                rse * self.setup.Mse \
                + self.setup.rep * self.setup.Mep \
                + self.setup.rpi * self.setup.Mpi \
                + self.setup.rir * self.setup.Mir \
                + self.setup.tau_p * self.setup.Mse_p \
                + self.setup.tau_i * self.setup.Mse_i
            qh = spsolve(self.setup.Imat - self.setup['h'] * MM.T, q0)
            q0 = qh
            if t >= self.setup['tint']:
                irm = 0.0

        for n in range(1, self.setup.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            self.prav[n-1] += s * q0[self.setup.s2i[n-1, s, e, p, i]]
            self.prav[n - 1] *= 1.0 /(n * self.setup.weights[n-1])
        
class WeakHouseholdIsolationModel(HouseholdModel):
    '''Weak household isolation model assumes a compliant percentage of
    househods will isolate if there is at least one symptomatic case'''
    def __init__(self, setup):
        super().__init__(setup)

    def solve(self):
        alpha_c = self.setup['npi']['compliance']
        epsilon = self.setup['npi']['global_reduction']

        # Overwrite the susceptible to exposed matrix
        Ise = array([],dtype=int32)
        Jse = array([],dtype=int32)
        Vse = array([])
        for n in range(1, self.setup.nmax + 1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            I = self.setup.s2i[n-1,s,e,p,i]
                            if (s>0):
                                Ise = append(Ise, I)
                                Jse = append(
                                    Jse, self.setup.s2i[n-1,s-1,e+1,p,i])
                                if (i==0):
                                    val = float(s)
                                else:
                                    val = (1.0 - alpha_c) * float(s)
                                Vse = append(Vse, val)
                                Ise = append(Ise, I)
                                Jse = append(Jse, I)
                                Vse = append(Vse, -val)  
        Mse = csr_matrix((Vse, (Ise, Jse)), self.setup.matrix_size)
            
        irm = 1.0
        q0 = zeros(self.setup.imax)
        for n in range(1,self.setup.nmax+1):
            q0[self.setup.s2i[n-1,n,0,0,0]] = self.setup.weights[n-1]
        for ti, t in enumerate(self.setup.trange):
            self.prev[ti] = (self.setup.ii @ q0)
            if t > 0:
                self.pdi[ti] = \
                    self.pdi[ti-1] + (alpha_c * self.prev[ti]) * self.setup['h']
                
            rse = \
                irm * self.setup['import_rate'] \
                + (1.0 - alpha_c) * (
                    self.setup.beta_p * (self.setup.pp @ q0)
                    + self.setup.beta_i*(self.prev[ti])) \
                + alpha_c * self.setup.beta_p * (self.setup.ptilde @ q0)
        
            if self.setup.NPI_active(t):
                rse = rse*(1.0 - epsilon)
            
            MM = \
                rse * Mse \
                + self.setup.rep * self.setup.Mep \
                + self.setup.rpi * self.setup.Mpi \
                + self.setup.rir * self.setup.Mir \
                + self.setup.tau_p * self.setup.Mse_p \
                + self.setup.tau_i * self.setup.Mse_i
            qh = spsolve(self.setup.Imat - self.setup['h'] * MM.T, q0)
            q0 = qh
            if t >= self.setup['tint']:
                irm = 0.0

        for n in range(1, self.setup.nmax+1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            self.prav[n-1] += s * q0[self.setup.s2i[n-1, s, e, p, i]]
            self.prav[n-1] *= 1.0 / (n * self.setup.weights[n-1])

class StrongHouseholdIsolationModelSetup(Setup):
    '''This object creates matrices for a household model with six indices:
    1. size         3. exposed      5. infected
    2. susceptible  4. prodromal    6. quarantined (yes/no)
    '''
    def __init__(self, params=DEFAULT_PARAMS):
        super().__init__(params)
        self.weights = params['pages'] / nsum(params['pages'])
        self.nmax = len(self.weights)
        self.nbar = self.weights @ arange(1, self.nmax + 1)

        self.s2i = zeros(
            (self.nmax+1, self.nmax+1, self.nmax+1, self.nmax+1, self.nmax+1, 2),
            dtype=int32)
        k = 0
        for n in range(1, self.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            for f in range(0,2):
                                self.s2i[n-1,s,e,p,i,f] = k
                                k += 1
        self.imax = k
        # Initialise the indices for sparse array generation

        Ise = array([], dtype=int32)
        Jse = array([], dtype=int32)
        Vse = array([])

        Ise_p = array([], dtype=int32)
        Jse_p = array([], dtype=int32)
        Vse_p = array([])

        Ise_i = array([],dtype=int32)
        Jse_i = array([],dtype=int32)
        Vse_i = array([])

        Iep = array([],dtype=int32)
        Jep = array([],dtype=int32)
        Vep = array([])

        Iir = array([],dtype=int32)
        Jir = array([],dtype=int32)
        Vir = array([])

        If = array([],dtype=int32)
        Jf = array([],dtype=int32)
        Vf = array([])

        self.ii = zeros(self.imax)
        self.jj = zeros(self.imax)
        self.pp = zeros(self.imax)
        self.ff = zeros(self.imax)
        self.ptilde = zeros(self.imax)
        for n in range(1, self.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            for f in range(0,2):
                                I = self.s2i[n-1, s, e, p, i, f]

                                if f==0:
                                    self.ii[I] = float(i)
                                    self.pp[I] = float(p)
                                    if (s>0):
                                        Ise = append(Ise,I)
                                        Jse = append(Jse, self.s2i[n-1,s-1,e+1,p,i,f])
                                        val = float(s)
                                        Vse = append(Vse,val)
                                        Ise = append(Ise,I)
                                        Jse = append(Jse,I)
                                        Vse = append(Vse,-val)
                                else:   # f==1
                                    self.ff[I] = float(n)                  
                                self.jj[I] = float(i)
                                        
                                if ((s>0) and (p>0)):
                                    Ise_p = append(Ise_p,I)
                                    Jse_p = append(Jse_p, self.s2i[n-1,s-1,e+1,p,i,f])
                                    val = float(s*p)/(float(n)**(-params['eta']))
                                    Vse_p = append(Vse_p,val)
                                    Ise_p = append(Ise_p,I)
                                    Jse_p = append(Jse_p,I)
                                    Vse_p = append(Vse_p,-val)      
                                if ((s>0) and (i>0)):
                                    Ise_i = append(Ise_i,I)
                                    Jse_i = append(Jse_i, self.s2i[n-1,s-1,e+1,p,i,f])
                                    val = float(s*i)/(float(n)**(-params['eta']))
                                    Vse_i = append(Vse_i,val)
                                    Ise_i = append(Ise_i,I)
                                    Jse_i = append(Jse_i,I)
                                    Vse_i = append(Vse_i,-val)                   
                                if (e>0):
                                    Iep = append(Iep,I)
                                    Jep = append(Jep, self.s2i[n-1,s,e-1,p+1,i,f])
                                    val = float(e)
                                    Vep = append(Vep,val)
                                    Iep = append(Iep,I)
                                    Jep = append(Jep,I)
                                    Vep = append(Vep,-val)
                                if (i>0):
                                    Iir = append(Iir,I)
                                    Jir = append(Jir,self.s2i[n-1,s,e,p,i-1,f])
                                    val = float(i)
                                    Vir = append(Vir,val)
                                    Iir = append(Iir,I)
                                    Jir = append(Jir,I)
                                    Vir = append(Vir,-val)
                                if (f==1):
                                    If = append(If,I)
                                    Jf = append(Jf,self.s2i[n-1,s,e,p,i,0])
                                    val = (1.0/14.0)
                                    Vf = append(Vf,val)
                                    If = append(If,I)
                                    Jf = append(Jf,I)
                                    Vf = append(Vf,-val)

        self.matrix_size = (self.imax, self.imax)
        self.Mse = csr_matrix((Vse, (Ise, Jse)), self.matrix_size)
        self.Mse_p = csr_matrix((Vse_p, (Ise_p, Jse_p)), self.matrix_size)
        self.Mse_i = csr_matrix((Vse_i, (Ise_i, Jse_i)), self.matrix_size)
        self.Mep = csr_matrix((Vep, (Iep, Jep)), self.matrix_size)
        self.Mir = csr_matrix((Vir, (Iir, Jir)), self.matrix_size)
        self.Mf = csr_matrix((Vf, (If, Jf)), self.matrix_size)

        self.rep = 1.0 / params['latent_period']
        self.rpi = 1.0 / params['prodrome_period']
        self.rir = 1.0 / params['infectious_period']
        self.beta_p = params['RGp'] / params['prodrome_period']
        self.beta_i = params['RGi'] / params['infectious_period']
        self.tau_p = (
            params['SAPp'] * self.rpi * (2.0**params['eta'])) \
            / (1.0 - params['SAPp'])
        self.tau_i = (
            params['SAPi'] * self.rir * (2.0**params['eta'])) \
            / (1.0-params['SAPi'])
        self.Imat = eye(*self.matrix_size)

class StrongHouseholdIsolationModel(HouseholdModel):
    '''Strong household isolation model assumes a compliant percentage of
    househods will isolate for a period of 14 days.'''
    def __init__(self, setup):
        super().__init__(setup)

    def solve(self):
        alpha_c = self.setup['npi']['compliance']
        epsilon = self.setup['npi']['global_reduction']
        Ipi = array([], dtype=int32)
        Jpi = array([], dtype=int32)
        Vpi = array([])
        for n in range(1, self.setup.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            for f in range(0,2):
                                I = self.setup.s2i[n-1,s,e,p,i,f]
                                if (p>0):
                                    Ipi = append(Ipi, I)
                                    if ((f==0) and (i==0) and ((s+e+p)==n)):
                                        Jpi = append(Jpi, self.setup.s2i[n-1,s,e,p-1,i+1,1])
                                        val = alpha_c*float(p)
                                        Vpi = append(Vpi,val)

                                        Ipi = append(Ipi,I)
                                        Jpi = append(Jpi,self.setup.s2i[n-1,s,e,p-1,i+1,0])
                                        val = (1.0 - alpha_c)*float(p)
                                        Vpi = append(Vpi,val)

                                        Ipi = append(Ipi,I)
                                        Jpi = append(Jpi,I)
                                        Vpi = append(Vpi,-float(p))
                                    else:
                                        Jpi = append(Jpi,self.setup.s2i[n-1,s,e,p-1,i+1,f])
                                        val = float(p)
                                        Vpi = append(Vpi,val)
                                        Ipi = append(Ipi,I)
                                        Jpi = append(Jpi,I)
                                        Vpi = append(Vpi,-val)
        Mpi = csr_matrix((Vpi, (Ipi, Jpi)), self.setup.matrix_size)
            
        irm = 1.0
        q0 = zeros(self.setup.imax)
        for n in range(1, self.setup.nmax+1):
            q0[self.setup.s2i[n-1,n,0,0,0,0]] = self.setup.weights[n-1]

        for ti, t in enumerate(self.setup.trange):
            self.prev[ti] = (self.setup.jj @ q0)
            if t > 0:
                self.pdi[ti] = \
                        self.pdi[ti-1] + (self.setup.ff @ q0) * self.setup['h']
                
            rse = \
                irm * self.setup['import_rate'] \
                + self.setup.beta_p * (self.setup.pp @ q0) \
                + self.setup.beta_i * (self.setup.ii @ q0)
        
            if self.setup.NPI_active(t):
                rse = rse * (1.0 - epsilon)
            
            MM = \
                rse * self.setup.Mse \
                + self.setup.rep * self.setup.Mep \
                + self.setup.rpi * Mpi \
                + self.setup.rir * self.setup.Mir \
                + self.setup.tau_p * self.setup.Mse_p \
                + self.setup.tau_i * self.setup.Mse_i \
                + self.setup.Mf
            qh = spsolve(self.setup.Imat - self.setup['h'] * MM.T, q0)
            q0 = qh
            if t >= self.setup['tint']:
                irm = 0.0

        for n in range(1, self.setup.nmax +1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            for f in range(0, 2):
                                self.prav[n-1] += s * q0[self.setup.s2i[n-1,s,e,p,i,f]]
            self.prav[n-1] *= 1.0 / (n * self.setup.weights[n-1])
