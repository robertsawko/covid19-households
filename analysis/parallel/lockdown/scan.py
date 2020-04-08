'''Run and pickle the result of a scan along the following model parameters:
    - global reduction (now parameterised across a range from the COMIX dataset)
    - compliance
    - secondary attack probabilities
    - end of NPIs
'''
from itertools import product
from copy import deepcopy
from numpy import array, linspace
from dill import dump
from pandas import DataFrame
from models.configs import CONFIG_WITH_DOUBLING_TIME as SPEC
from models.parallel import ParallelExecutionToHDF5

if __name__ == '__main__':
    comply_range = linspace(0.65,0.8,4)
    globred_range = linspace(0.65,0.8,4)
    lockdown_durations = array([21.0, 90.0, 180.0])
    SAPi_range = array([0.4, 0.6, 0.8, 0.9])

    doe = DataFrame(
        list(product(
            comply_range,
            globred_range,
            lockdown_durations,
            SAPi_range)),
        columns = [
            'Compliance',
            'Global reduction',
            'Lockdown duration',
            'SAPi'])

    def record2spec(record, spec):
        p2i = 0.5
        sapi = record['SAPi']
        spec['SAPi'] = sapi
        spec['SAPp'] = p2i * sapi
        spec['npi']['end'] = spec['npi']['start'] + record['Lockdown duration']
        spec['final_time'] = spec['npi']['start'] + record['Lockdown duration']
        spec['npi']['compliance'] = record['Compliance']
        spec['npi']['global_reduction'] = record['Global reduction']
        return spec

    SPEC['npi']['type'] = 'weak'
    my_parallel_run = ParallelExecutionToHDF5(
        SPEC, doe, record2spec, 'lockdown.h5')
    my_parallel_run.execute()
