'''Run and pickle the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from copy import deepcopy
from numpy import array, linspace
from dill import dump
from models.utils import setup_from_dict, from_setup
from models.configs import DEFAULT_PARAMS

if __name__ == '__main__':
    comply_range = linspace(0.0, 1.0, 6)
    globred_range = array([0.0, 0.25, 0.5, 0.75])
    npi_types = ['individual', 'weak', 'strong']

    msg = 'Done global reduction range {0:d} of {1:d} and compliange range {2:d} of {3:d}'
    configs=[]
    for npi_type npi_types:
        new_config = deepcopy(DEFAULT_PARAMS)
        new_config['npi']['type'] = npi_type
        configs.append(new_config)
    setups = {c['npi']['type']: setup_from_dict(c) for c in configs}
    results = {s: [] for s in setups.keys()}

    for ig, g in enumerate(globred_range):
        partial_results = {s: [] for s in setups.keys()}
        for ic, c in enumerate(comply_range):
            for isolation in setups.keys():
                m = from_setup(setups[isolation])
                m.setup['npi']['compliance'] = c
                m.setup['npi']['global_reduction'] = g
                m.solve()
                partial_results[isolation].append(m)
            print(msg.format(ig + 1, len(globred_range), ic + 1, len(comply_range)))
        for isolation in setups.keys():
            results[isolation].append(partial_results[isolation])

    suffix_and_result = [
        ('', results['individual']),
        ('_weak', results['weak']),
        ('_strong', results['strong'])]
    for suffix, results in suffix_and_result:
        with open('outputs{0}.pkl'.format(suffix), 'wb') as f:
            dump(results, f)

