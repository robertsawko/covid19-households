'''Run and pickle the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from numpy import array, linspace
from dill import dump
from models.household import BasicModelSetup, IndividualIsolationModel
from models.household import WeakHouseholdIsolationModel

if __name__ == '__main__':
    comply_range = linspace(0.0, 1.0, 6)
    globred_range = array([0.0, 0.25, 0.5, 0.75])

    msg = 'Done global reduction range {0:d} of {1:d} and compliange range {2:d} of {3:d}'
    setup = BasicModelSetup()
    individual = []
    weak = []

    for ig, g in enumerate(globred_range):
        individual_gr = []
        weak_gr = []
        for ic, c in enumerate(comply_range):
            individual_gr.append(IndividualIsolationModel(setup, g, c))
            weak_gr.append(WeakHouseholdIsolationModel(setup, g, c))
            print(msg.format(ig + 1, len(globred_range), ic + 1, len(comply_range)))
        individual.append(individual_gr)
        weak.append(weak_gr)

    with open('outputs.pkl', 'wb') as f:
        dump(individual, f)
    with open('outputs_weak.pkl', 'wb') as f:
        dump(weak, f)
