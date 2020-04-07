'''Plot the result of a scan from an HDF5 file
'''
from numpy import ceil, ones, zeros
from h5py import File
from pandas import read_hdf
from numpy import max as nmax
from matplotlib.pyplot import subplots
from dill import load

class TimeSeries:
    def __init__(self, idx, hdf_file):
        self.prev = hdf_file['run{0:03d}'.format(idx) + '/prev'][:]
        self.time = hdf_file['run{0:03d}'.format(idx) + '/time'][:]

if __name__ == '__main__':
    file_name = 'outputs.hdf5'
    doe = read_hdf(file_name, key='design_of_experiment')
    f = File(file_name, 'r')

    npi_types = list(doe.groupby('Type').count().index)
    global_reductions = list(doe.groupby('Global reduction').count().index)
    fig, axis = subplots(
        len(global_reductions), len(npi_types), figsize=(8, 3.75))

    unmitigated = TimeSeries(0, f)
    for idx_im, im in enumerate(npi_types):
        model_subset = doe[doe['Type'] == im]
        for idx_g, g in enumerate(global_reductions):
            axes = axis[idx_g, idx_im]
            model_gr_subset = model_subset[model_subset['Global reduction'] == g]
            indices = list(
                model_gr_subset.sort_values(by='Compliance').index)
            time_series = [TimeSeries(i, f) for i in indices]
            for ts in reversed(time_series[1:]):
                axes.plot(
                    ts.time, ts.prev)
                    # label='{:.0f}% compliance'.format(
                        # 100*m.setup['npi']['compliance']),
                    # colour=[0.5, 0.5, 0.5+(0.5*m.setup['npi']['compliance'])])
            baseline = time_series[0]
            # axes.title.set_text('Global Reduction {:.0f}%'.format(
                # 100*baseline.setup['npi']['global_reduction']))
            # import pdb
            # pdb.set_trace()
            axes.plot(
                baseline.time, baseline.prev,
                label='Baseline',
                color=[0, 0, 0])
            #yup = 1.05 * unmitigated.peak_value
            #npi_start, npi_end = baseline.setup['npi']['start'], baseline.setup['npi']['end']
            #axes.plot([npi_start, npi_start], [0, yup], ls='--', c='k')
            #axes.plot([npi_end, npi_end], [0, yup], ls='--', c='k')
            #axes.set_xlim([0, baseline.setup['final_time']])
            #axes.set_ylim([0, yup])
            axes.set_xlabel('Time (days)')
            axes.set_ylabel('Number of Cases')
                
    fig.tight_layout()
    fig.savefig('./time_series_all.pdf')
    f.close()


