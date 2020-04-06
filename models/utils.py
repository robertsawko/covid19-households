from numpy import array
from json import dump, load
from models.configs import DEFAULT_PARAMS
from models.household import IndividualIsolationModel
from models.household import WeakHouseholdIsolationModel
from models.household import StrongHouseholdIsolationModel

def from_dict(spec):
    '''Create a household model from a Python dictionary'''
    type_to_constructor = {
        'individual': IndividualIsolationModel,
        'weak': WeakHouseholdIsolationModel,
        'strong': StrongHouseholdIsolationModel
    }
    return type_to_constructor[spec['npi']['type']](spec)

def from_json(fname):
    '''Create a household model from a JSON file'''
    with open(fname, 'w') as f:
        spec = load(fname)
        return from_dict(spec)

if __name__ == '__main__':
    with open('params.json', 'w') as f:
        dump(DEFAULT_PARAMS, f, indent=4)
