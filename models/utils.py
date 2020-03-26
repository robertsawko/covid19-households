from numpy import array
from json import dump

DEFAULT_PARAMS = {
    #Interpretable parameters:
    'latent_period': 5.0,       # Days in E class before becoming infectious
    'prodrome_period': 3.0,     # Days infectious but not symptomatic
    'infectious_period': 4.0,   # Days infectious and symptomatic
    'RGp': 0.5,                 # Contribution to R0 outside the household from pre-symptomatic
    'RGi': 1.0,                 # Contribution to R0 outside the household during symptomatic
    'eta': 0.8,                 # Parameter of the Cauchemez model: HH transmission ~ n^(-eta)
    'import_rate': 0.001,       # International importation rate
    'Npop': 5.6e7,              # Total population
    'SAPp': 0.4,                # Secondary attack probability for a two-person household with one susceptible and one prodrome
    'SAPi': 0.8,                # Secondary attack probability for a two-person household with one susceptible and one infective
    'final_time': 180.0,        # Final time [days]
    'h': 0.04,                  # Time step [days] - 0.04 is approximately an hour timestep for time integration
    'tint': 10.0,               # Time at which we neglect imports
    # Population information
    'pages': [                  # From 2001 UK census the percentages of households from size 1 to 8 are:
            30.28, 34.07, 15.51, 13.32, 4.88, 1.41, 0.33, 0.20],
    # Characterisation of non-pharmaceutical interventions (NPI)
    'npi_start': 21,            # Start of NPIs
    'npi_end': 42,              # End of NPIs
    'compliance': 0.0,          # Percentage of the population compliant with the policy
    'global_reduction': 0.0,    # Global reduction
}

if __name__ == '__main__':
    with open('params.json', 'w') as f:
        dump(DEFAULT_PARAMS, f, indent=4)
