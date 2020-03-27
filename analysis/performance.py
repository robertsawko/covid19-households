'''To evaluate performance computational performance of each model under default
conditions.
'''
from time import time
from numpy import mean
from models.household import BasicModelSetup, IndividualIsolationModel
from models.household import WeakHouseholdIsolationModel
from models.household import StrongHouseholdIsolationModelSetup
from models.household import StrongHouseholdIsolationModel
from models.configs import DEFAULT_PARAMS

class Test:
    def __init__(self, setup_class, model_class, repeats=1):
        self.repeats = repeats
        self.setup_class = setup_class
        self.model_class = model_class

        measurements = []
        for _ in range(repeats):
            start = time()
            setup = setup_class()
            end = time()
            measurements.append(end-start)
        self.average_setup_time = mean(measurements)
        measurements = []
        for _ in range(repeats):
            start = time()
            model = model_class(setup)
            model.solve()
            end = time()
            measurements.append(end-start)
        self.average_solve_time = mean(measurements)
        self.no_of_variables = setup.imax
    
    def __repr__(self):
        return '''
Setup class: {4:s}
Model class: {5:s}
No of variables: {0}
Average setup time [s]: {1:f}
Average solve time [s]: {2:f}
No of repeats: {3:d}'''.format(
        self.no_of_variables,
        self.average_setup_time,
        self.average_solve_time,
        self.repeats,
        self.setup_class.__name__,
        self.model_class.__name__)


if __name__ == '__main__':
    repeats = 10
    individual = Test(BasicModelSetup, IndividualIsolationModel, repeats)
    weak = Test(BasicModelSetup, WeakHouseholdIsolationModel, repeats)
    strong = Test(
        StrongHouseholdIsolationModelSetup,
        StrongHouseholdIsolationModel,
        repeats)

    for test in [individual, weak, strong]:
        print(test)
