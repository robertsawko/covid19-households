'''To evaluate performance computational performance of each model under default
conditions.
'''
from copy import deepcopy
from time import time
from numpy import mean
from models.utils import from_dict
from models.configs import DEFAULT_PARAMS

class Test:
    def __init__(self, npi_type, repeats=1):
        self.repeats = repeats
        config = deepcopy(DEFAULT_PARAMS)
        config['npi']['type'] = npi_type

        measurements = []
        for _ in range(repeats):
            start = time()
            model = from_dict(config)
            end = time()
            measurements.append(end-start)
        self.average_setup_time = mean(measurements)
        measurements = []
        for _ in range(repeats):
            start = time()
            model.solve()
            end = time()
            measurements.append(end-start)
        self.average_solve_time = mean(measurements)
        self.no_of_variables = model.imax
        self.model_class_name = type(model).__name__
    
    def __repr__(self):
        return '''
Model class: {4:s}
No of variables: {0}
Average setup time [s]: {1:f}
Average solve time [s]: {2:f}
No of repeats: {3:d}'''.format(
        self.no_of_variables,
        self.average_setup_time,
        self.average_solve_time,
        self.repeats,
        self.model_class_name)


if __name__ == '__main__':
    repeats = 1
    npis = ['individual', 'weak', 'strong']
    tests = [Test(npi, repeats=repeats) for npi in npis]
    for test in tests:
        print(test)
