'''Design of experiment. This file is used to generate the experiment matrix.

Currently using 4 parameters:
    - compliance
    - global reduction
    - type
    - duration
'''
from itertools import product
from numpy import linspace, arange
from pandas import DataFrame

if __name__ == '__main__':
    compliances = linspace(0, 1, 6)                 # Population compliance
    global_reductions = linspace(0.0, 0.75, 4)      # Efficiency of NPIs
    npi_types = ['individual', 'weak', 'strong']    # Type of NPIs
    durations = 7 * arange(1, 5)                    # Duration of NPIs

    doe = DataFrame(
        list(product(
            compliances,
            global_reductions,
            npi_types,
            durations)),
        columns = ['Compliance', 'Global reduction', 'Type', 'Duration'])
    doe.to_pickle('doe.pkl')
