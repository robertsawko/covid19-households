# covid-households

Modelling Household Isolation for COVID-19.

Content:

 * `analysis` full implementation of the models in a notebook form:
    * `scan_compliance_and_reduction` sensitivity analysis of all three
      isolation models with for a uniform grid of global reduction and
      compliance,
    * `performance.py` some simple tests to run different types of the model,
    * `parallel` demonstration on how to run large batches in parallel with
      HPC. Two examples:
      * maual pickling with `dill`
      * compliance and reduction scan with HDF5
 * `manuscript` LaTeX write-up of the model equations
 * `models` modular implementation of the models (work-in-progress)
 * `notebooks` full implementation of the models in a notebook form. These
   serve as prototypes for later *production* implemenntation.

## Prerequisites

Currently requiring the following non-standard python packages

```
numpy
scipy
dill
pandas
h5py
tables
```

`h5py` requires a parallel installation of HDF5. If such an installation exists
it `h5py` can be installed with the following line:

```bash
CC="mpicc" HDF5_MPI="ON" HDF5_DIR="/path/to/parallel-hdf5" \
  pip install --no-binary==h5py h5py
```
