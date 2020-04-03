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
        npi_start, npi_end = \
            baseline.spec['npi']['start'], baseline.spec['npi']['end']
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
                colour=[0.3, 0.3, 0.3 + (0.7 * m.spec['npi']['compliance'])])
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

    fig, axis = subplots(1, 2, figsize=(8, 3.75))
    comply_range = [m.spec['npi']['compliance'] for m in gr_runs[0]]
    for ig, models in enumerate(gr_runs):
        axis[0].plot(
            comply_range,
            percred[ig,:],
            label='{:.0f}%'.format(
                100*models[0].spec['npi']['global_reduction']))
    axis[0].set_xlabel('Compliance with Isolation')
    axis[0].set_ylabel('Percentage of Baseline peak')
    axis[0].title.set_text('Individual isolation: Mitigation')
    axis[0].set_xlim([0, 1])
    axis[0].set_ylim([0, 100])
    axis[0].legend(title="Global Reduction")
    for ig, models in enumerate(gr_runs):
        axis[1].plot(comply_range, persdays[ig, :])
    axis[1].set_xlabel('Compliance with Isolation')
    axis[1].set_ylabel('Number of Person-Days Isolation')
    axis[1].title.set_text('Individual isolation: Cost')
    axis[1].set_xlim([0, 1])
    axis[1].set_ylim([0, ceil(nmax(persdays))])
    fig.tight_layout()
    fig.savefig('./mit_costs{0}.pdf'.format(suffix))

    fig, axis = subplots(2, 4, figsize=(10,4))
    for n in range(1, unmitigated.nmax+1):
        axes = axis[(n - 1)//4, (n - 1)%4]
        for models in gr_runs:
            axes.plot(
                comply_range,
                [m.prav[n-1] for m in models],
                label='{:.0f}%'.format(100*models[0].spec['npi']['global_reduction']))
            axes.set_xlim([0, 1.0])
            axes.set_ylim([0,0.5])
            axes.set_xlabel('Compliance')
            axes.set_ylabel('Prob(Avoid)')
            axes.title.set_text('Household size {0:d}'.format(int(n)))
        if n==nmax:
            fig.legend(
                bbox_to_anchor=(1.04,1),
                loc="upper left",
                title="Global Reduction")
    fig.tight_layout()
    fig.savefig('prob_avoid{0}.pdf'.format(suffix))

if __name__ == '__main__':
    suffixes = ['', '_weak', '_strong']
    for s in suffixes:
        plot_from_dill(s)
