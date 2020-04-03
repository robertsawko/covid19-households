from itertools import product
from numpy import arange, linspace
from pandas import DataFrame
from models.parallel import ParallelExecutionToHDF5

if __name__ == '__main__':
    compliances = linspace(0, 1, 2)                 # Population compliance
    global_reductions = linspace(0.0, 0.75, 2)      # Efficiency of NPIs
    npi_types = ['individual', 'weak', 'strong']    # Type of NPIs
    durations = 7 * arange(2, 5)                    # Duration of NPIs

    doe = DataFrame(
        list(product(
            compliances,
            global_reductions,
            npi_types,
            durations)),
        columns = ['Compliance', 'Global reduction', 'Type', 'Duration'])

    my_parallel_run = ParallelExecutionToHDF5(doe, 'test.hdf5')
    my_parallel_run.execute()
