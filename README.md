# Hadfit
`hadfit` is a Python library build to help analysing correlation functions extracted from Lattice
QCD simulations. The goal of the library is to provide a solid ground from which one can develop 
further tools. In addition, the library contains a set of classes to extract the ground mass of 
observables using a multi-state variational regression technique; this method is implemented inside
`msfit`. Although `hadfit` is tailored for mesonic correlation functions, it can be easily expanded
to baryonic data.

A simple script that utilises `hadfit` to extract the ground mass of different mesonic correlation
functions is contained inside `scripts/`. Inside `scripts/`, a `.tar` file is included. This file
contains a sample for reproducibility. Due to the huge size of the correlation function data, only 
one example is included. For more examples, email `sergiozteskate@gmail.com`.

## How to install the library.
The library and its dependencies are manages by `poetry`; one can install `poetry` using the following
command:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
```

After succesfully installing `poetry`, one can work in a virtual environment using:

```bash
poetry install; poetry shell
```

This command will install all the needed dependencies and spawn a virtual environment where `hadfit` is
installed. In the case in which one wanted to install `hadfit` globally in the current machine, one can
use instead:

```bash
poetry build; cd dist; pip install *.tar.gz
```

## Modules contained in hadfit:
`hadfit` contains different modules that can be used at will.

 - *hadron*: Contains some classes to load and manipulate hadronic correlation functions.
 - *model*:  Model definition; wrapper around `lmfit.Model` that includes automatic
   computation of Jacobians and symbolic expression manipulation.
 - *msfit*:  Multistate fit regression algorithm used to extract ground masses from mesonic
   correlation functions. Can be expanded to baryons easily.
 - *analysis*: Tools to analyse and manipulate the data extracted from the regression procedure. It
   also contains tools to post-process Fastum data.

---
Sergio Chaves García-Mascaraque; e-mail: `sergiozte@gmail.com`.
