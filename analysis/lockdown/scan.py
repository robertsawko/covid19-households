'''Run and pickle the result of a scan along the following model parameters:
    - global reduction (now parameterised across a range from the COMIX dataset)
    - compliance
    - secondary attack probabilities
    - end of NPIs
'''
from copy import deepcopy
from numpy import array, linspace
from dill import dump
from models.utils import from_dict
from models.configs import CONFIG_WITH_DOUBLING_TIME as SPEC

if __name__ == '__main__':
    comply_range = linspace(0.65,0.8,4)
    globred_range = linspace(0.65,0.8,4)
    lockdown_durations = array([21.0, 90.0, 180.0])
    SAPi_range = array([0.4, 0.6, 0.8, 0.9])

    msg = 'Done global reduction range {0:d} of {1:d}; compliance range {2:d} of {3:d}; duration range {4:d} of {5:d}; SAP range {6:d} of {7:d}.'
    results = []

    for ig, g in enumerate(globred_range):
        partial_results = []
        for ic, c in enumerate(comply_range):
            for ild, ld in enumerate(lockdown_durations):
                for isap, sap in enumerate(SAPi_range):
                    spec = deepcopy(SPEC)
                    p2i = spec['SAPp']/spec['SAPi']
                    spec['SAPi'] = sap
                    spec['SAPp'] = p2i*sap
                    spec['npi']['end'] = (spec['npi']['start'] + ld)
                    spec['final_time'] = (spec['npi']['start'] + lockdown_durations[-1])
                    spec['npi']['type'] = 'weak'
                    spec['npi']['compliance'] = c
                    spec['npi']['global_reduction'] = g
                    m = from_dict(spec)
                    m.solve()
                    partial_results.append(m)
                print(msg.format(
                    ig + 1, len(globred_range),
                    ic + 1, len(comply_range),
                    ild + 1, len(lockdown_range),
                    isap + 1, len(SAPi_range)))
        results.append(partial_results)

    with open('ld-outputs.pkl', 'wb') as f:
        dump(results, f)
