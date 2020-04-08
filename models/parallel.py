'''Parallel toolkit'''
from os.path import join
from copy import deepcopy
from mpi4py import MPI
from h5py import File, Empty
from models.utils import from_dict

class ParallelExecutionToHDF5:
    '''This class serves as a context for a parallel exectuion which saves to
    an HDF5 file'''
    def __init__(
            self,
            base_spec,
            design_of_experiment,
            record2spec,
            output_file_name):

        self.design_of_experiment = design_of_experiment
        self.file_name = output_file_name
        self.base_spec = base_spec
        # This will be a pattern for keys representing different outputs
        self.key_pattern_fmt = 'run{0:03d}'
        self.specs = [deepcopy(base_spec) for _ in design_of_experiment.itertuples()]

        for idx, spec in enumerate(self.specs):
            self.specs[idx] = record2spec(
                design_of_experiment.loc[idx], spec)

    def _initialise_hdf5_structure(self, f):
        for idx, spec in enumerate(self.specs):
            group = f.create_group(self.key_pattern_fmt.format(idx))
            no_of_time_steps = int(spec['final_time'] / spec['h'])
            group.create_dataset(
                'time',
                shape=(no_of_time_steps,),
                dtype='f', )
            group.create_dataset(
                'prev',
                shape=(no_of_time_steps,),
                dtype='f', )
            group.attrs.create('Npop', spec['Npop'])
            group.create_dataset('nbar', shape=(1,), dtype='f')
            # TODO: add more variables/arrays here

    def _store_computed_vars(self, f, idx, model):
        group = f[self.key_pattern_fmt.format(idx)]
        group['nbar'][:] = model.nbar
        group['prev'][:] = model.prev
        group['time'][:] = model.trange
        # TODO: add more variables/arrays here

    def execute(self):
        comm = MPI.COMM_WORLD
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

        for idx in range(rank, len(self.specs), comm.size):
            print('Process {0:d}/{1:d} will process index {2}/{3}'.format(
                rank, comm.size, idx, len(self.design_of_experiment)))
            model = from_dict(self.specs[idx])
            model.solve()
            self._store_computed_vars(f, idx, model)


        print('Process {0}/{1} at the barrier'.format(rank, comm.size))
        comm.barrier()
        f.close()

