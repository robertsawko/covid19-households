'''Plot the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from numpy import zeros, ones
from numpy import max as nmax
from matplotlib.pyplot import subplots
from models.baseline import BaselineModel, ModelSetup
from pickle import load

with open('outputs.pkl', 'rb') as f:
    gr_runs = load(f)

no_of_global_reductions = len(gr_runs)
no_of_compliances = len(gr_runs)
percred = 100 * ones((no_of_global_reductions, no_of_compliances))
persdays = zeros((no_of_global_reductions, no_of_compliances))

fig, axis = subplots(
    no_of_global_reductions, 2,
    figsize=(9, 2.5 * no_of_global_reductions))

for ig, runs in enumerate(gr_runs):
    axes = axis[ig, 0]
    for ic, m in enumerate(runs[-1:0:-1]):
        m.plot_cases(
            axes,
            label='{:.0f}% compliance'.format(100*m.compliance),
            colour=[0.5, 0.5, 0.5+(0.5*m.compliance)])
        # percred[ig, ic] = 100.0*(nmax(prev[g,c,:])/nmax(prev[0,0,:]))
    baseline = runs[0]
    baseline.plot_cases(
        axes,
        label='Baseline',
        colour=[0, 0, 0])
    yup = 1.05 * baseline.peak_value
    # Draw intervention lines
    npi_begin, npi_end = baseline.setup.dist_start, baseline.setup.dist_end
    axes.plot([npi_begin, npi_begin], [0, yup], ls='--', c='k')
    axes.plot([npi_end, npi_end], [0,yup], ls='--', c='k')

    axes.set_xlim([0, baseline.setup.trange[-1]])
    axes.set_ylim([0, yup])
    axes.set_xlabel('Time (days)')
    axes.set_ylabel('Number of Cases')
    axes.title.set_text('Global Reduction {:.0f}%'.format(
        100*baseline.global_reduction))
    axes = axis[ig, 1]
    for ic, m in enumerate(runs[-1:0:-1]):
        m.plot_person_days_of_isolation(
            axes,
            label='{:.0f}%'.format(100*m.compliance),
            colour=[0.3, 0.3, 0.3 + (0.7*m.compliance)])

        # persdays[g,c] = (Npop/nbar)*pdi[g,c,-2]
    baseline.plot_person_days_of_isolation(
        axes,
        label='Baseline',
        colour=[0, 0, 0])
    axes.set_xlim([0, baseline.setup.trange[-1]])
    
    axes.set_ylim([0, 1.05*runs[-1].max_person_days_of_isolation])
    if (ig==0):
        axes.legend(bbox_to_anchor=(1.04, 1), loc='upper left', title='Compliance')                                                          
    axes.set_xlabel('Time (days)')
    axes.set_ylabel('Person-Days Isolation')
    axes.title.set_text('Global Reduction {:.0f}%'.format(100*baseline.global_reduction))
fig.tight_layout()
fig.savefig('./time_series.pdf')
