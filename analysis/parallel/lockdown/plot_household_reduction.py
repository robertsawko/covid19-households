'''Plot the result of a scan from an HDF5 file
'''
from numpy import arange, array, ceil, ones, zeros, zeros_like
from h5py import File
from pandas import read_hdf
from numpy import max as nmax
from matplotlib.pyplot import figure
from matplotlib.pyplot import style
from matplotlib.ticker import FixedLocator
#style.use('ggplot')

class States:
    def __init__(self, idx, hdf_file):
        runid = 'run{0:03d}'.format(idx)
        states = runid + '/states'
        self.s2i = hdf_file[states + '/index2state'][:]
        self.at_lockdown = hdf_file[states + '/at_lockdown'][:]
        self.max = array([
            hdf_file[states + '/{0}d'.format(n)][:]
            for n in arange(42, 21+180, 21, dtype=int)]).max(axis=0)
        self.Npop = hdf_file[runid].attrs['Npop']
        self.nbar = hdf_file[runid + '/nbar'][:]

    @property
    def cases(self):
        return self.Npop / self.nbar * self.prev

    def difference(self, househould_size):
        n = househould_size
        y = zeros(n+1)
        # z = np.zeros(n+1)
        for s in range(0,n+1):
            for e in range(0,n+1-s):
                for p in range(0,n+1-s-e):
                    for i in range(0,n+1-s-e-p):
                        y[i] += \
                            (self.Npop / self.nbar) \
                            * self.max[int(self.s2i[n-1,s,e,p,i])]
                        y[i] -= \
                            (self.Npop / self.nbar) \
                            * self.at_lockdown[int(self.s2i[n-1,s,e,p,i])]
        return y

if __name__ == '__main__':
    file_name = 'lockdown.h5'
    doe = read_hdf(file_name, key='design_of_experiment')
    f = File(file_name, 'r')

    fig = figure(
        constrained_layout=True, figsize=(8,8))
    gs = fig.add_gridspec(4, 4)
    y1 = [0, 0, 0, 1, 1, 2, 2, 3]
    x1 = [0, 1, 2, 0, 2, 0, 2, 0]
    x2 = [1, 2, 4, 2, 4, 2, 4, 4]
    axis = [
        fig.add_subplot(gs[y1[n-1], x1[n-1]:x2[n-1]])
        for n in range(1, 8+1)]
    lockdowns = list(doe.groupby('Lockdown duration').count().index)
    for idx_l, l in enumerate(lockdowns):
        model_subset = doe[doe['Lockdown duration'] == l]
        model_indices = list(model_subset.index)
        states = [States(i, f) for i in model_indices]
        for n in range(1, 8+1):
            differences = array([s.difference(n) for s in states])
            print(differences.mean(axis=0))
            print(differences.std(axis=0))

            #print(differences.mean(axis=0))
            x = arange(1,n+1)
            axis[n-1].bar(
                x+idx_l*0.25-0.25,
                differences.mean(axis=0)[1:],
                0.25,
                label='{0:d} d'.format(int(l)))


    axis[-1].legend(loc='upper right', title='Lockdown duration')
    for i, ax in enumerate(axis):
        ax.set_title('Size {0:d}'.format(i+1))
        ax.xaxis.set_major_locator(FixedLocator(arange(1, i+2)))
    fig.savefig('a.png', tight_layout=True, format='png', dpi=300)
    f.close()


