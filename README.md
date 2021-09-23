# Hadfit
`hadfit` is a library to help analysis correlation functions extracted from lattice QCD
calculations. The goal of the library is to provide a solid ground from which one can develop
further tools. In addition, it contains some modules tailored for spectroscopy of lattice QCD
correlation functions. `hadfit` contains an algoritm to extract ground masses from mesonic
correlation functions. Although `hadfit` is tailored for mesonic correlation functions, it can
easily be expanded to baryonic data.

A simple script that utilises `hadfit` to obtain the ground mass of different mesonic correlation
functions is contained inside `scripts/`. Some real mesonic correlation function data is contained
inside `scripts/data` to test the algorithm.

## The modules:
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
