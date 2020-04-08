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
    file_name = 'lockdown.h5'
    doe = read_hdf(file_name, key='design_of_experiment')
    f = File(file_name, 'r')

    durations = list(doe.groupby('Lockdown duration').count().index)
    global_reductions = list(doe.groupby('Global reduction').count().index)
    fig, axis = subplots(
        len(global_reductions), len(durations),
        figsize=(8, 8),
        sharey=True)

    for idx_d, d in enumerate(durations):
        model_subset = doe[doe['Lockdown duration'] == d]
        for idx_g, g in enumerate(global_reductions):
            axes = axis[idx_g, idx_d]
            model_gr_subset = model_subset[model_subset['Global reduction'] == g]
            indices = list(
                model_gr_subset.sort_values(by='Compliance').index)
            time_series = [TimeSeries(i, f) for i in indices]
            for idx_c, ts in enumerate(time_series):
                compliance = model_gr_subset.loc[indices[idx_c]]['Compliance']
                axes.plot(
                    ts.time, ts.prev,
                    label='{:.0f}% compliance'.format(
                        100 * compliance),
                    color=[0.5, 0.5, 0.5+0.5*(compliance-0.65)/(0.8-0.65)])
            # baseline = time_series[0]
            axes.title.set_text('Global Reduction {:.0f}%'.format(
                100*model_gr_subset.iloc[0]['Global reduction']))
            #axes.plot(
                #baseline.time, baseline.prev,
                #label='Baseline',
                #color=[0, 0, 0])
            #yup = 1.05 * unmitigated.peak_value
            #npi_start, npi_end = baseline.setup['npi']['start'], baseline.setup['npi']['end']
            #axes.plot([npi_start, npi_start], [0, yup], ls='--', c='k')
            #axes.plot([npi_end, npi_end], [0, yup], ls='--', c='k')
            #axes.set_xlim([0, baseline.setup['final_time']])
            #axes.set_ylim([0, yup])
            axes.set_xlabel('Time (days)')
            axes.set_ylabel('Number of Cases')
                
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        ncol=4, loc='lower center', bbox_to_anchor=(0.5,0.0))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    fig.savefig('./lockdown.pdf')
    f.close()


