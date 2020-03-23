'''Plot the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from numpy import zeros, ones
from numpy import max as nmax
from matplotlib.pyplot import subplots
#from models.baseline import (
#        lg, lc, trange, prev, nbar, Npop, comply_range,
#        globred_range, tend, dist_start, dist_end, pdi)
from pickle import load

with open('outputs.pkl', 'rb') as f:
    lg, lc, trange, prev, nbar, Npop, comply_range, globred_range, tend, dist_start, dist_end, pdi = load(f)

percred = 100 * ones((lg, lc))
persdays = zeros((lg, lc))

fig, axis = subplots(lg, 2, figsize=(9, 2.5 * lg))
for g in range(lg):
    axes = axis[g, 0]
    for c in range(lc-1,0,-1):
        axes.plot(
            trange,
            (Npop/nbar)*prev[g,c,:],
            label= '{:.0f}% compliance'.format(100*comply_range[c]),
            c=[0.5, 0.5, 0.5+(0.5*comply_range[c])])
        percred[g,c] = 100.0*(nmax(prev[g,c,:])/nmax(prev[0,0,:]))
    axes.plot(
        trange,
        (Npop/nbar)*prev[g,0,:],
        label='Baseline',
        c=[0, 0, 0])
    yup = 1.05 * (Npop / nbar) * nmax(prev[0, 0, :])
    axes.plot([dist_start, dist_start], [0, yup], ls='--', c='k')
    axes.plot([dist_end, dist_end], [0,yup], ls='--', c='k')
    axes.set_xlim([0, trange[tend-1]])
    axes.set_ylim([0,yup])
    axes.set_xlabel('Time (days)')
    axes.set_ylabel('Number of Cases')
    axes.title.set_text('Global Reduction {:.0f}%'.format(100*globred_range[g]))
    axes = axis[g, 1]
    for c in range(lc-1,0,-1):
        axes.plot(
            trange,
            (Npop / nbar) * pdi[g, c, :],
            label= '{:.0f}%'.format(100*comply_range[c]),
            c=[0.3, 0.3, 0.3 + (0.7*comply_range[c])])
        persdays[g,c] = (Npop/nbar)*pdi[g,c,-2]
    axes.plot(
        trange,
        (Npop / nbar) * pdi[g, 0, :],
        label='Baseline',c=[0, 0, 0])
    axes.set_xlim([0, trange[tend-1]])
    axes.set_ylim([0, 1.05*nmax((Npop / nbar) * pdi[g, lc-1, :])])
    if (g==0):
        axes.legend(bbox_to_anchor=(1.04, 1), loc='upper left', title='Compliance')                                                          
    axes.set_xlabel('Time (days)')
    axes.set_ylabel('Person-Days Isolation')
    axes.title.set_text('Global Reduction {:.0f}%'.format(100*globred_range[g]))
fig.tight_layout()
fig.savefig('./time_series.pdf')
