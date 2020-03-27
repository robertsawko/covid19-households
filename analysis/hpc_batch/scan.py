''' This is showing how we're running a larger batch

Note: some experiments within a design share identical setup. Currently, I am
not accounting for this.
'''
from copy import deepcopy
from argparse import ArgumentParser
from pickle import load
from dill import dump
from models.utils import from_dict
from multiprocessing import Pool
from models.configs import DEFAULT_PARAMS

def run_simulation(configuration):
    model = from_dict(configuration)
    model.solve()
    return model

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--experiment_file',
        default='doe.pkl',
        type=str,
        help='Filenames of the pickled dataframe with the DOE')
    parser.add_argument(
        '--sim0', default=0, type=int, help='ID of the first simulation')
    parser.add_argument(
        '--pool_size', default=4, type=int, help='Size of multiprocessing pool')
    parser.add_argument(
        '--num_of_simulations',
        default=1,
        type=int,
        help='Number of simulations')

    args = parser.parse_args()
    
    with open(args.experiment_file, 'rb') as f:
        doe = load(f)

    sim0 = args.sim0
    num_of_simulations = args.num_of_simulations
    pool = Pool(args.pool_size)
    configs = [deepcopy(DEFAULT_PARAMS) for _ in range(num_of_simulations)]
    for i, c in enumerate(configs):
        row = doe.loc[sim0 + i]
        c['npi']['compliance'] = row.Compliance
        c['npi']['global_reduction'] = row['Global reduction']
        c['npi']['type'] = row.Type
        c['npi']['end'] = c['npi']['start'] + row.Duration
    result = pool.map_async(run_simulation, configs).get()
    pool.close()
    with open('out{0:04d}.pkl'.format(sim0), 'wb') as f:
        dump(result, f)
