'''Parallel toolkit'''
from os.path import join
from copy import deepcopy
from mpi4py import MPI
from h5py import File
from models.configs import DEFAULT_PARAMS as SPEC
from models.utils import from_dict

def _make_spec(record):
    spec = deepcopy(SPEC)
    spec['npi']['compliance'] = record['Compliance']
    spec['npi']['global_reduction'] = record['Global reduction']
    spec['npi']['type'] = record['Type']
    spec['npi']['end'] = spec['npi']['start'] + record['Duration']
    return spec

class ParallelExecutionToHDF5:
    '''This class serves as a context for a parallel exectuion which saves to
    an HDF5 file'''
    def __init__(self, design_of_experiment, output_file_name):
        self.design_of_experiment = design_of_experiment
        self.file_name = output_file_name
        # This will be a pattern for keys representing different outputs
        self.key_pattern_fmt = 'run{0:03d}'

    def _initialise_hdf5_structure(self, f):
        no_of_time_steps = int(SPEC['final_time'] / SPEC['h'])
        for idx in range(0, len(self.design_of_experiment)):
            group = f.create_group(self.key_pattern_fmt.format(idx))
            group.create_dataset(
                'time',
                shape=(no_of_time_steps,),
                dtype='f', )
            group.create_dataset(
                'prev',
                shape=(no_of_time_steps,),
                dtype='f', )

    def execute(self):
        comm = MPI.COMM_WORLD
        size = comm.size
        rank = comm.rank
        if rank == 0:
            self.design_of_experiment.to_hdf(
                self.file_name,
                key='design_of_experiment',
                mode='w')
        comm.barrier()

        f = File(self.file_name, 'r+', driver='mpio', comm=comm)
        # Structure needs to be modified _collectively_.
        self._initialise_hdf5_structure(f)

        for idx in range(rank, len(self.design_of_experiment), size):
            print('Process {0:d}/{1:d} will process index {2}/{3}'.format(
                rank, size, idx, len(self.design_of_experiment)))
            record = self.design_of_experiment.loc[idx]
            spec = _make_spec(record) 
            model = from_dict(spec)
            model.solve()
            f[join(self.key_pattern_fmt.format(idx), 'prev')][:] = model.prev
            f[join(self.key_pattern_fmt.format(idx), 'time')][:] = model.trange

        print('Process {0}/{1} at the barrier'.format(rank, size))
        comm.barrier()
        f.close()

