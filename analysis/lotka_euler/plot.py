'''Plot the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from numpy import ceil, ones, zeros
from numpy import max as nmax
from matplotlib.pyplot import subplots
from dill import load

def plot_from_dill(suffix=''):
    with open('outputs{0}.pkl'.format(suffix), 'rb') as f:
        gr_runs = load(f)

    no_of_global_reductions = len(gr_runs)
    no_of_compliances = len(gr_runs[0])
    percred = 100 * ones((no_of_global_reductions, no_of_compliances))
    persdays = zeros((no_of_global_reductions, no_of_compliances))

    fig, axis = subplots(
        no_of_global_reductions, 2,
        figsize=(9, 2.5 * no_of_global_reductions))

    unmitigated = gr_runs[0][0]
    for ig, models in enumerate(gr_runs):
        axes = axis[ig, 0]
        for ic in range(len(models)-1, 0, -1):
            m = models[ic]
            m.plot_cases(
                axes,
                label='{:.0f}% compliance'.format(100*m.spec['npi']['compliance']),
                colour=[0.5, 0.5, 0.5+(0.5*m.spec['npi']['compliance'])])
            percred[ig, ic] = 100.0*m.peak_ratio(unmitigated)
        baseline = models[0]
        baseline.plot_cases(
            axes,
            label='Baseline',
            colour=[0, 0, 0])
        yup = 1.05 * unmitigated.peak_value
        # Draw intervention lines
        npi_start, npi_end = baseline.spec['npi']['start'], baseline.spec['npi']['end']
        axes.plot([npi_start, npi_start], [0, yup], ls='--', c='k')
        axes.plot([npi_end, npi_end], [0, yup], ls='--', c='k')

        axes.set_xlim([0, baseline.spec['final_time']])
        axes.set_ylim([0, yup])
        axes.set_xlabel('Time (days)')
        axes.set_ylabel('Number of Cases')
        axes.title.set_text('Global Reduction {:.0f}%'.format(
            100*baseline.spec['npi']['global_reduction']))
        axes = axis[ig, 1]
        for ic in range(len(models) - 1, 0, -1):
            m=models[ic]
            m.plot_person_days_of_isolation(
                axes,
                label='{:.0f}%'.format(100 * m.spec['npi']['compliance']),
                colour=[0.3, 0.3, 0.3 + (0.7*m.spec['npi']['compliance'])])
            persdays[ig, ic] = m.persdays
        baseline.plot_person_days_of_isolation(
            axes,
            label='Baseline',
            colour=[0, 0, 0])
        axes.set_xlim([0, baseline.spec['final_time']])
        
        axes.set_ylim([0, 1.05*models[-1].max_person_days_of_isolation])
        if (ig==0):
            axes.legend(bbox_to_anchor=(1.04, 1), loc='upper left', title='Compliance')                                                          
        axes.set_xlabel('Time (days)')
        axes.set_ylabel('Person-Days Isolation')
        axes.title.set_text('Global Reduction {:.0f}%'.format(
            100*baseline.spec['npi']['global_reduction']))
    fig.tight_layout()
    fig.savefig('./time_series{0}.pdf'.format(suffix))

if __name__ == '__main__':
    suffixes = ['', '_weak']
    for s in suffixes:
        plot_from_dill(s)
