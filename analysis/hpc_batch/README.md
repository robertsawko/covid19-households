# HPC batching

This will execute in parallel multiple runs of the model with different params.

First a design of experiment file has to be created with a `doe.py`.
Subsequently we need to execute `scan.py` file multiple times to process the
full design.

TODO: how to store the outputs?
