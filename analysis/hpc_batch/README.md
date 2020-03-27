# HPC batching

This will execute in parallel multiple runs of the model with different values
of the parameters.

First a design of experiment (DOE) file has to be created with a `doe.py`.
Subsequently, we need to execute `scan.py` file multiple times to process the
full design. The latter is now a command-line interface and can execute a
number of simulation from a DOE starting with specified ID.

TODO: how to store the outputs?

Currently the outputs of each batch are stored as serialised list using `dill`
module.
