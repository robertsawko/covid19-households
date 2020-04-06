'''Run and pickle the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from copy import deepcopy
from numpy import array, linspace
from dill import dump
from models.utils import from_dict
from models.configs import DEFAULT_PARAMS as SPEC

if __name__ == '__main__':
    comply_range = linspace(0.0, 1.0, 6)
    globred_range = array([0.0, 0.25, 0.5, 0.75])
    npi_types = ['individual', 'weak', 'strong']

    msg = 'Done global reduction range {0:d} of {1:d} and compliange range {2:d} of {3:d}'
    results = {s: [] for s in npi_types}

    for ig, g in enumerate(globred_range):
        partial_results = {s: [] for s in npi_types}
        for ic, c in enumerate(comply_range):
            for npi_type in npi_types:
                spec = deepcopy(SPEC)
                spec['npi']['type'] = npi_type
                spec['npi']['compliance'] = c
                spec['npi']['global_reduction'] = g
                model = from_dict(spec)
                model.solve()
                partial_results[npi_type].append(model)
            print(msg.format(ig + 1, len(globred_range), ic + 1, len(comply_range)))
        for npi_type in npi_types:
            results[npi_type].append(partial_results[npi_type])

    suffix_and_result = [
        ('', results['individual']),
        ('_weak', results['weak']),
        ('_strong', results['strong'])]
    for suffix, results in suffix_and_result:
        with open('outputs{0}.pkl'.format(suffix), 'wb') as f:
            dump(results, f)

