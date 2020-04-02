'''The household isolation models for COVID-19 epidemic in the UK. All models
assume:
    * non-pharmaceutical interventions (NPI) of length T.

Notation of the essential variables follows the manuscript to improve the
readability.

Note: some repetition is allowed for readability e.g. epsilon and alpha_c
assignment are repeated in every solve method.
'''
from abc import ABC, abstractmethod
from copy import deepcopy
from numpy import append, arange, array, int32, log, zeros, zeros_like
from numpy import sum as nsum
from numpy import max as nmax
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from models.configs import DEFAULT_PARAMS

class ModelBuilder(ABC):
    def build(self, spec):
        '''Build a model according to model specification'''
        model = self.class_name(spec)
        model.weights = array(spec['pages']) / nsum(spec['pages'])
        model.nmax = len(model.weights)
        model.nbar = model.weights @ arange(1, model.nmax + 1)
        model.prav = zeros(model.nmax)       # Probability of avoiding by household size
        self.create_matrices(spec, model)
        return model

    @abstractmethod
    def create_matrices(self, spec, model):
        pass

class FiveClassModelBuilder(ModelBuilder):
    '''This needs to be a hidden class to abstract commonoalities between
    Weak and Individual models. The classes are:
    1. size         3. exposed      5. infected
    2. susceptible  4. prodromal
    '''
    def create_common_matrices(self, spec, model):
        # A mapping of multi-index onto a linear index.
        model.s2i = zeros(5*(model.nmax+1,), dtype=int32)

        k=0
        for n in range(1, model.nmax+1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            model.s2i[n-1, s, e, p, i] = k
                            k += 1
        model.imax = k

        # Initialise the indices for sparse array generation
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

        model.ii = zeros(model.imax)
        model.pp = zeros(model.imax)
        model.ptilde = zeros(model.imax)

        for n in range(1, model.nmax + 1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            I = model.s2i[n-1, s, e, p, i]
                            model.ii[I] = float(i)
                            model.pp[I] = float(p)
                            
                            if (s > 0) and (p > 0):
                                Ise_p = append(Ise_p, I)
                                Jse_p = append(Jse_p, model.s2i[n-1,s-1,e+1,p,i])
                                val = float(s*p)/(float(n)**(-spec['eta'])) # CAUCHEMEZ MODEL
                                Vse_p = append(Vse_p, val)
                                Ise_p = append(Ise_p, I)
                                Jse_p = append(Jse_p, I)
                                Vse_p = append(Vse_p, -val)      
                            if (s > 0) and (i > 0):
                                Ise_i = append(Ise_i, I)
                                Jse_i = append(Jse_i, model.s2i[n-1,s-1,e+1,p,i])
                                val = float(s*i)/(float(n)**(-spec['eta'])) # CAUCHEMEZ MODEL
                                Vse_i = append(Vse_i, val)
                                Ise_i = append(Ise_i, I)
                                Jse_i = append(Jse_i, I)
                                Vse_i = append(Vse_i,-val)                   
                            if e > 0:
                                Iep = append(Iep, I)
                                Jep = append(Jep, model.s2i[n-1,s,e-1,p+1,i])
                                val = float(e)
                                Vep = append(Vep,val)
                                Iep = append(Iep,I)
                                Jep = append(Jep,I)
                                Vep = append(Vep,-val)
                            if i > 0:
                                Iir = append(Iir, I)
                                Jir = append(Jir, model.s2i[n-1,s,e,p,i-1])
                                val = float(i)
                                Vir = append(Vir,val)
                                Iir = append(Iir,I)
                                Jir = append(Jir,I)
                                Vir = append(Vir,-val)
                            if p > 0:
                                Ipi = append(Ipi, I)
                                Jpi = append(Jpi, model.s2i[n-1,s,e,p-1,i+1])
                                val = float(p)
                                Vpi = append(Vpi,val)
                                Ipi = append(Ipi,I)
                                Jpi = append(Jpi,I)
                                Vpi = append(Vpi,-val)

        model.matrix_size = (model.imax, model.imax)
        model.Mse_p = csr_matrix((Vse_p, (Ise_p, Jse_p)), model.matrix_size)
        model.Mse_i = csr_matrix((Vse_i, (Ise_i, Jse_i)), model.matrix_size)
        model.Mep = csr_matrix((Vep, (Iep, Jep)), model.matrix_size)
        model.Mpi = csr_matrix((Vpi, (Ipi, Jpi)), model.matrix_size)
        model.Mir = csr_matrix((Vir, (Iir, Jir)), model.matrix_size)

class IndividualIsolationModelBuilder(FiveClassModelBuilder):
    def __init__(self):
        self.class_name = IndividualIsolationModel

    def create_matrices(self, spec, model):
        self.create_common_matrices(spec, model)

        Ise = array([], dtype=int32)
        Jse = array([], dtype=int32)
        Vse = array([])
        for n in range(1, model.nmax + 1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            I = model.s2i[n-1, s, e, p, i]
                            if s > 0:
                                Ise = append(Ise, I)
                                Jse = append(Jse, model.s2i[n-1, s-1, e+1, p, i])
                                val = float(s)
                                Vse = append(Vse, val)
                                Ise = append(Ise, I)
                                Jse = append(Jse, I)
                                Vse = append(Vse, -val)   
        model.Mse = csr_matrix((Vse, (Ise, Jse)), model.matrix_size)

class WeakHouseholdIsolationModelBuilder(FiveClassModelBuilder):
    def __init__(self):
        self.class_name = WeakHouseholdIsolationModel

    def create_matrices(self, spec, model):
        self.create_common_matrices(spec, model)

        Ise = array([], dtype=int32)
        Jse = array([], dtype=int32)
        Vse = array([])
        model.ptilde = zeros(model.imax)
        for n in range(1, model.nmax + 1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            I = model.s2i[n-1, s, e, p, i]
                            if i==0:
                                model.ptilde[I] = float(p)
                            if s > 0:
                                Ise = append(Ise, I)
                                Jse = append(
                                    Jse, model.s2i[n-1,s-1,e+1,p,i])
                                if (i==0):
                                    val = float(s)
                                else:
                                    val = (1.0 - spec['npi']['compliance']) * float(s)
                                Vse = append(Vse, val)
                                Ise = append(Ise, I)
                                Jse = append(Jse, I)
                                Vse = append(Vse, -val)  
        model.Mse = csr_matrix((Vse, (Ise, Jse)), model.matrix_size)


class StrongHouseholdIsolationModelBuilder(ModelBuilder):
    '''This object creates matrices for a household model with six classes:
    1. size         3. exposed      5. infected
    2. susceptible  4. prodromal    6. quarantined (yes/no)
    '''
    def __init__(self):
        self.class_name = StrongHouseholdIsolationModel

    def create_matrices(self, spec, model):
        model.s2i = zeros(6*(model.nmax+1,), dtype=int32)
        k = 0
        for n in range(1, model.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            for f in range(0,2):
                                model.s2i[n-1,s,e,p,i,f] = k
                                k += 1
        model.imax = k
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

        Ipi = array([], dtype=int32)
        Jpi = array([], dtype=int32)
        Vpi = array([])

        model.ii = zeros(model.imax)
        model.jj = zeros(model.imax)
        model.pp = zeros(model.imax)
        model.ff = zeros(model.imax)
        model.ptilde = zeros(model.imax)

        for n in range(1, model.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            for f in range(0,2):
                                I = model.s2i[n-1, s, e, p, i, f]

                                if f==0:
                                    model.ii[I] = float(i)
                                    model.pp[I] = float(p)
                                    if (s>0):
                                        Ise = append(Ise,I)
                                        Jse = append(Jse, model.s2i[n-1,s-1,e+1,p,i,f])
                                        val = float(s)
                                        Vse = append(Vse,val)
                                        Ise = append(Ise,I)
                                        Jse = append(Jse,I)
                                        Vse = append(Vse,-val)
                                else:   # f==1
                                    model.ff[I] = float(n)                  
                                model.jj[I] = float(i)
                                        
                                if ((s>0) and (p>0)):
                                    Ise_p = append(Ise_p,I)
                                    Jse_p = append(Jse_p, model.s2i[n-1,s-1,e+1,p,i,f])
                                    val = float(s*p)/(float(n)**(-spec['eta']))
                                    Vse_p = append(Vse_p,val)
                                    Ise_p = append(Ise_p,I)
                                    Jse_p = append(Jse_p,I)
                                    Vse_p = append(Vse_p,-val)      
                                if ((s>0) and (i>0)):
                                    Ise_i = append(Ise_i,I)
                                    Jse_i = append(Jse_i, model.s2i[n-1,s-1,e+1,p,i,f])
                                    val = float(s*i)/(float(n)**(-spec['eta']))
                                    Vse_i = append(Vse_i,val)
                                    Ise_i = append(Ise_i,I)
                                    Jse_i = append(Jse_i,I)
                                    Vse_i = append(Vse_i,-val)                   
                                if (e>0):
                                    Iep = append(Iep,I)
                                    Jep = append(Jep, model.s2i[n-1,s,e-1,p+1,i,f])
                                    val = float(e)
                                    Vep = append(Vep,val)
                                    Iep = append(Iep,I)
                                    Jep = append(Jep,I)
                                    Vep = append(Vep,-val)
                                if (i>0):
                                    Iir = append(Iir,I)
                                    Jir = append(Jir,model.s2i[n-1,s,e,p,i-1,f])
                                    val = float(i)
                                    Vir = append(Vir,val)
                                    Iir = append(Iir,I)
                                    Jir = append(Jir,I)
                                    Vir = append(Vir,-val)
                                if (f==1):
                                    If = append(If,I)
                                    Jf = append(Jf,model.s2i[n-1,s,e,p,i,0])
                                    val = (1.0/14.0)
                                    Vf = append(Vf,val)
                                    If = append(If,I)
                                    Jf = append(Jf,I)
                                    Vf = append(Vf,-val)
                                if (p>0):
                                    Ipi = append(Ipi, I)
                                    if ((f==0) and (i==0) and ((s+e+p)==n)):
                                        Jpi = append(Jpi, model.s2i[n-1,s,e,p-1,i+1,1])
                                        val = spec['npi']['compliance']*float(p)
                                        Vpi = append(Vpi,val)

                                        Ipi = append(Ipi,I)
                                        Jpi = append(Jpi, model.s2i[n-1,s,e,p-1,i+1,0])
                                        val = (1.0 - spec['npi']['compliance'])*float(p)
                                        Vpi = append(Vpi,val)

                                        Ipi = append(Ipi,I)
                                        Jpi = append(Jpi,I)
                                        Vpi = append(Vpi,-float(p))
                                    else:
                                        Jpi = append(Jpi, model.s2i[n-1,s,e,p-1,i+1,f])
                                        val = float(p)
                                        Vpi = append(Vpi,val)
                                        Ipi = append(Ipi,I)
                                        Jpi = append(Jpi,I)
                                        Vpi = append(Vpi,-val)

        model.matrix_size = (model.imax, model.imax)
        model.Mse = csr_matrix((Vse, (Ise, Jse)), model.matrix_size)
        model.Mse_p = csr_matrix((Vse_p, (Ise_p, Jse_p)), model.matrix_size)
        model.Mse_i = csr_matrix((Vse_i, (Ise_i, Jse_i)), model.matrix_size)
        model.Mep = csr_matrix((Vep, (Iep, Jep)), model.matrix_size)
        model.Mir = csr_matrix((Vir, (Iir, Jir)), model.matrix_size)
        model.Mf = csr_matrix((Vf, (If, Jf)), model.matrix_size)
        model.Mpi = csr_matrix((Vpi, (Ipi, Jpi)), model.matrix_size)

class Setup(ABC):
    def rescale_beta(self):
        '''Solution of the Lotka-Euler equation to fix global transmission
        from doubling time'''
        ww = zeros(self.imax)
        cc = []
        for n in range(1, self.nmax + 1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            I = self.s2i[n-1, s, e, p, i]

                            if ((e==1) and (s==(n-1))):
                                ww[I] = n * self.weights[n-1]

                            if not ((e==0) and (p==0) and (i==0)):
                                cc.append(I)
        rr = log(2.0)/self.params['doubling_time']
        MM = \
            self.rep * self.Mep \
            + self.rpi * self.Mpi \
            + self.rir * self.Mir \
            + self.tau_p * self.Mse_p \
            + self.tau_i * self.Mse_i \
            - rr * Imat
        Qc = -(MM[:, cc][cc, :])
        z = spsolve(
            Qc,
            self.beta_p * self.pp[cc] + self.beta_i * self.ii[cc])

        beta_scale = 1.0 / (z@ww[cc])

        self.beta_i *= beta_scale
        self.beta_p *= beta_scale

class HouseholdModel(ABC):
    def __init__(self, setup):
        self.setup = deepcopy(setup)
        self.trange = arange(
            0, setup['final_time'], setup['h'])

        self.prev = zeros_like(self.trange)
        self.pdi = zeros_like(self.trange)  # Person-days in isolation

        self.rep = 1.0 / setup['latent_period']
        self.rpi = 1.0 / setup['prodrome_period']
        self.rir = 1.0 / setup['infectious_period']
        self.beta_p = setup['RGp'] / setup['prodrome_period']
        self.beta_i = setup['RGi'] / setup['infectious_period']
        self.tau_p = (
            setup['SAPp'] * self.rpi * (2.0**self.setup['eta'])) \
            / (1.0 - setup['SAPp'])
        self.tau_i = (
            setup['SAPi'] * self.rir * (2.0**setup['eta'])) \
            / (1.0 - self.setup['SAPi'])

    def _initialise_common_vars(self):
        return \
            self.setup['npi']['compliance'], \
            self.setup['npi']['global_reduction'], \
            eye(*self.matrix_size), \
            self.setup['import_rate'], \
            self.setup['h']


    def plot_cases(self, axes, label, colour):
        axes.plot(
            self.trange,
            (self.setup['Npop'] / self.nbar) * self.prev,
            label=label,
            c=colour)

    def peak_ratio(self, other):
        return nmax(self.prev)/nmax(other.prev)

    @property
    def peak_value(self):
        return (self.setup['Npop'] / self.nbar) * nmax(self.prev)

    @property
    def persdays(self):
        return (self.setup['Npop'] / self.nbar) * self.pdi[-2]

    @property
    def max_person_days_of_isolation(self):
        return (self.setup['Npop'] / self.nbar) * nmax(self.pdi)

    def plot_person_days_of_isolation(self, axes, label, colour):
        axes.plot(
            self.trange,
            (self.setup['Npop'] / self.nbar) * self.pdi,
            label=label,
            c=colour)

    def NPI_active(self, t):
        '''Check if non-pharmaceutical interventions are active at time t'''
        return self.setup['npi']['start'] <= t <= self.setup['npi']['end']

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
        alpha_c, epsilon, Imat, Lambda, h = self._initialise_common_vars()

        irm = 1.0
        q0 = zeros(self.imax)
        for n in range(1, self.nmax + 1):
            q0[self.s2i[n-1, n, 0, 0, 0]] = self.weights[n - 1]
        for ti, t in enumerate(self.trange):
            self.prev[ti] = (self.ii @ q0)
            if ti > 0:
                self.pdi[ti] = \
                    self.pdi[ti-1] + (alpha_c * self.prev[ti]) * h
            rse = \
                irm * Lambda \
                + self.beta_p * (self.pp @ q0) \
                + (1.0 - alpha_c) * self.beta_i * self.prev[ti]

            if self.NPI_active(t):
                rse = rse * (1.0 - epsilon)

            MM = \
                rse * self.Mse \
                + self.rep * self.Mep \
                + self.rpi * self.Mpi \
                + self.rir * self.Mir \
                + self.tau_p * self.Mse_p \
                + self.tau_i * self.Mse_i
            qh = spsolve(Imat - h * MM.T, q0)
            q0 = qh
            if t >= self.setup['tint']:
                irm = 0.0

        for n in range(1, self.nmax+1):
            for s in range(0,n+1):
                for e in range(0,n+1-s):
                    for p in range(0,n+1-s-e):
                        for i in range(0,n+1-s-e-p):
                            self.prav[n-1] += s * q0[self.s2i[n-1, s, e, p, i]]
            self.prav[n - 1] *= 1.0 /(n * self.weights[n-1])
        
class WeakHouseholdIsolationModel(HouseholdModel):
    '''Weak household isolation model assumes a compliant percentage of
    househods will isolate if there is at least one symptomatic case'''
    def __init__(self, setup):
        super().__init__(setup)

    def solve(self):
        alpha_c, epsilon, Imat, Lambda, h = self._initialise_common_vars()

        irm = 1.0
        q0 = zeros(self.imax)
        for n in range(1, self.nmax+1):
            q0[self.s2i[n-1, n, 0, 0, 0]] = self.weights[n-1]
        for ti, t in enumerate(self.trange):
            self.prev[ti] = (self.ii @ q0)
            if t > 0:
                self.pdi[ti] = \
                    self.pdi[ti-1] + (alpha_c * self.prev[ti]) * h
                
            rse = \
                irm * Lambda \
                + (1.0 - alpha_c) * (
                    self.beta_p * (self.pp @ q0)
                    + self.beta_i * (self.prev[ti])) \
                + alpha_c * self.beta_p * (self.ptilde @ q0)
        
            if self.NPI_active(t):
                rse = rse * (1.0 - epsilon)
            
            MM = \
                rse * self.Mse \
                + self.rep * self.Mep \
                + self.rpi * self.Mpi \
                + self.rir * self.Mir \
                + self.tau_p * self.Mse_p \
                + self.tau_i * self.Mse_i
            qh = spsolve(Imat - h * MM.T, q0)
            q0 = qh
            if t >= self.setup['tint']:
                irm = 0.0

        for n in range(1, self.nmax+1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            self.prav[n-1] += s * q0[self.s2i[n-1, s, e, p, i]]
            self.prav[n-1] *= 1.0 / (n * self.weights[n-1])

class StrongHouseholdIsolationModel(HouseholdModel):
    '''Strong household isolation model assumes a compliant percentage of
    househods will isolate for a period of 14 days.'''
    def __init__(self, setup):
        super().__init__(setup)

    def solve(self):
        alpha_c, epsilon, Imat, Lambda, h = self._initialise_common_vars()

        irm = 1.0
        q0 = zeros(self.imax)
        for n in range(1, self.nmax+1):
            q0[self.s2i[n-1,n,0,0,0,0]] = self.weights[n-1]

        for ti, t in enumerate(self.trange):
            self.prev[ti] = (self.jj @ q0)
            if t > 0:
                self.pdi[ti] = self.pdi[ti-1] + (self.ff @ q0) * h
                
            rse = \
                irm * Lambda \
                + self.beta_p * (self.pp @ q0) \
                + self.beta_i * (self.ii @ q0)
        
            if self.NPI_active(t):
                rse = rse * (1.0 - epsilon)
            
            MM = \
                rse * self.Mse \
                + self.rep * self.Mep \
                + self.rpi * self.Mpi \
                + self.rir * self.Mir \
                + self.tau_p * self.Mse_p \
                + self.tau_i * self.Mse_i \
                + self.Mf
            qh = spsolve(Imat - h * MM.T, q0)
            q0 = qh
            if t >= self.setup['tint']:
                irm = 0.0

        for n in range(1, self.nmax +1):
            for s in range(0, n+1):
                for e in range(0, n+1-s):
                    for p in range(0, n+1-s-e):
                        for i in range(0, n+1-s-e-p):
                            for f in range(0, 2):
                                self.prav[n-1] += s * q0[self.s2i[n-1,s,e,p,i,f]]
            self.prav[n-1] *= 1.0 / (n * self.weights[n-1])
