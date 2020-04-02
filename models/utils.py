from numpy import array
from json import dump, load
from models.configs import DEFAULT_PARAMS
from models.household import IndividualIsolationModelBuilder
from models.household import WeakHouseholdIsolationModelBuilder
from models.household import StrongHouseholdIsolationModelBuilder

def from_dict(spec):
    '''Create a household model from a Python dictionary'''
    type_to_builder = {
        'individual': IndividualIsolationModelBuilder(),
        'weak': WeakHouseholdIsolationModelBuilder(),
        'strong': StrongHouseholdIsolationModelBuilder()
    }
    return type_to_builder[spec['npi']['type']].build(spec)

def from_json(fname):
    '''Create a household model from a JSON file'''
    with open(fname, 'w') as f:
        spec = load(fname)
        return from_dict(spec)

if __name__ == '__main__':
    with open('params.json', 'w') as f:
        dump(DEFAULT_PARAMS, f, indent=4)
