# covid-households

Modelling Household Isolation for COVID-19.

Content:

 * `analysis` full implementation of the models in a notebook form:
    * `scan_compliance_and_reduction` sensitivity analysis of all three
      isolation models with for a uniform grid of global reduction and
      compliance,
    * `performance.py` some simple tests to run different types of the model,
    * `hpc_batch` demonstration on how to run large batches in parallel with
      HPC.
 * `manuscript` LaTeX write-up
 * `models` modular implementation of the models (work-in-progress)
 * `notebooks` full implementation of the models in a notebook form.

## Prerequisites

Currently requiring the following non-standard python packages

```
dill
pandas
```
