'''Run and pickle the result of a scan along the following model parameters:
    - global reduction 
    - compliance
'''
from numpy import array, linspace
from pickle import dump
from models.baseline import BaselineModel, BasicModelSetup

if __name__ == '__main__':
    comply_range = linspace(0.0, 1.0, 6)
    globred_range = array([0.0, 0.25, 0.5, 0.75])

    msg = 'Done global reduction range {0:d} of {1:d} and compliange range {2:d} of {3:d}'
    setup = BasicModelSetup()
    models = []
    for ig, g in enumerate(globred_range):
        compliance_runs = []
        for ic, c in enumerate(comply_range):
            compliance_runs.append(BaselineModel(setup, g, c))
            print(msg.format(ig + 1, len(globred_range), ic + 1, len(comply_range)))
        models.append(compliance_runs)

    with open('outputs.pkl', 'wb') as f:
        dump(models, f)

