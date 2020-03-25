'''Run and pickle the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from numpy import array, linspace
from dill import dump
from models.household import BasicModelSetup, IndividualIsolationModel
from models.household import WeakHouseholdIsolationModel
from models.household import StrongHouseholdIsolationModelSetup
from models.household import StrongHouseholdIsolationModel

if __name__ == '__main__':
    comply_range = linspace(0.0, 1.0, 6)
    globred_range = array([0.0, 0.25, 0.5, 0.75])

    msg = 'Done global reduction range {0:d} of {1:d} and compliange range {2:d} of {3:d}'
    setup = BasicModelSetup()
    strong_setup = StrongHouseholdIsolationModelSetup()
    individual = []
    weak = []
    strong = []

    for ig, g in enumerate(globred_range):
        individual_gr = []
        weak_gr = []
        strong_gr = []
        for ic, c in enumerate(comply_range):
            individual_gr.append(IndividualIsolationModel(setup, g, c))
            weak_gr.append(WeakHouseholdIsolationModel(setup, g, c))
            strong_gr.append(StrongHouseholdIsolationModel(
                strong_setup, g, c))
            print(msg.format(ig + 1, len(globred_range), ic + 1, len(comply_range)))
        individual.append(individual_gr)
        weak.append(weak_gr)
        strong.append(strong_gr)

    suffix_and_result = [
        ('', individual),
        ('_weak', weak),
        ('_strong', strong)]
    for suffix, results in suffix_and_result:
        with open('outputs{0}.pkl'.format(suffix), 'wb') as f:
            dump(results, f)

