'''This demonstrates how to generate design of experiment and execute a
parallel run whilst storing results in HDF5'''
from itertools import product
from numpy import arange, linspace
from pandas import DataFrame
from models.configs import DEFAULT_PARAMS as SPEC
from models.parallel import ParallelExecutionToHDF5

if __name__ == '__main__':
    compliances = linspace(0.0, 1.0, 6)             # Population compliance
    global_reductions = linspace(0.0, 1.0, 6)       # Efficiency of NPIs
    npi_types = ['individual', 'weak', 'strong']    # Type of NPIs
    durations = 7 * arange(2, 4)                    # Duration of NPIs

    doe = DataFrame(
        list(product(
            compliances,
            global_reductions,
            npi_types,
            durations)),
        columns = ['Compliance', 'Global reduction', 'Type', 'Duration'])

    my_parallel_run = ParallelExecutionToHDF5(SPEC, doe, 'outputs.h5')
    my_parallel_run.execute()
