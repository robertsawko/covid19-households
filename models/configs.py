'''Module containing fixed model configurations'''

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
    'npi': {
        'type': 'individual',   # Type e.g. individual isolation or weak home isolation
        'start': 21,            # Start of NPIs
        'end': 42,              # End of NPIs
        'compliance': 0.0,          # Percentage of the population compliant with the policy
        'global_reduction': 0.0,    # Global reduction
    }
}

CONFIG_WITH_DOUBLING_TIME = {
    #Interpretable parameters:
    'latent_period': 5.0,       # Days in E class before becoming infectious
    'prodrome_period': 3.0,     # Days infectious but not symptomatic
    'infectious_period': 4.0,   # Days infectious and symptomatic
    'doubling_time': 3.0,       # Doubling time in the absence of interventions
    'RGp': 0.5,                 # Contribution to R0 outside the household from pre-symptomatic
    'RGi': 1.0,                 # Contribution to R0 outside the household during symptomatic
    'eta': 0.8,                 # Parameter of the Cauchemez model: HH transmission ~ n^(-eta)
    'Npop': 6.7e7,              # Total population (This is UK at present)
    'SAPp': 0.4,                # Secondary attack probability for a two-person household with one susceptible and one prodrome
    'SAPi': 0.8,                # Secondary attack probability for a two-person household with one susceptible and one infective
    'final_time': 180.0,        # Final time [days]
    'h': 0.04,                  # Time step [days] - 0.04 is approximately an hour timestep for time integration
    'import_rate': 0.0001,      # International importation rate
    'tint': 0.01,               # Time at which we neglect imports
    # Population information
    'pages': [                  # From 2001 UK census the percentages of households from size 1 to 8 are:
            30.28, 34.07, 15.51, 13.32, 4.88, 1.41, 0.33, 0.20],
    # Characterisation of non-pharmaceutical interventions (NPI)
    'npi': {
        'type': 'individual',   # Type e.g. individual isolation or weak home isolation
        'start': 40,            # Start of NPIs
        'end': 180,             # End of NPIs
        'compliance': 0.0,      # Percentage of the population compliant with the policy
        'global_reduction': 0.0,# Global reduction
    }
}
