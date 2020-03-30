from numpy import array
from json import dump, load
from models.configs import DEFAULT_PARAMS
from models.household import BasicModelSetup
from models.household import StrongHouseholdIsolationModelSetup
from models.household import IndividualIsolationModel
from models.household import WeakHouseholdIsolationModel
from models.household import StrongHouseholdIsolationModel

def setup_from_dict(setup_dictionary):
    '''Get a setup class for a household model using Python dictionary'''
    type_to_contstructor = {
        'individual': BasicModelSetup,
        'weak': BasicModelSetup,
        'strong': StrongHouseholdIsolationModelSetup
    }
    return type_to_contstructor[setup_dictionary['npi']['type']](
        setup_dictionary)

def from_setup(setup):
    '''Create a household model from a Python dictionary and existing setup'''
    type_to_contstructor = {
        'individual': IndividualIsolationModel,
        'weak': WeakHouseholdIsolationModel,
        'strong': StrongHouseholdIsolationModel
    }
    return type_to_contstructor[setup.params['npi']['type']](setup)

def from_dict(setup_dictionary):
    '''Create a household model from a Python dictionary'''
    setup = setup_from_dict(setup_dictionary)
    return from_setup(setup)

def from_json(fname):
    '''Create a household model from a JSON file'''
    with open(fname, 'w') as f:
        setup_dictionary = load(fname)
        return from_dict(setup_dictionary)

if __name__ == '__main__':
    with open('params.json', 'w') as f:
        dump(DEFAULT_PARAMS, f, indent=4)
