# Hadfit

`hadfit` is a library to perform regressions on hadronic correlation functions extracted from
lattice QCD calculations. The library can be used to manipulate hadronic correlation functions and
fit them to multistate models. `hadfit` is specially tailored for mesonic correlation functions but
can be expanded to baryons easily.

A simple script that utilises `hadfit` to obtain the ground mass of different mesonic correlation
functions is contained inside `scripts/`.

## The modules:
`hadfit` contains different modules that can be used at will.

 - *hadron*: Contains some classes to load and manipulate hadronic correlation functions.
 - *model*:  Model definition. It is a wrapper around `lmfit.Model` that includes automatic
   computation of Jacobians and symbolic expressions manipulations.
 - *msfit*:  Multistate fit regression algorithm used to extract ground masses from mesonic
   correlation functions. Can be expanded to baryons easily.
 - *analysis*: Tools to analyse and manipulate the data extracted from the regression procedure. It
   also contains tools to post-process Fastum data.

---
Sergio Chaves García-Mascaraque; `sergiozte@gmail.com`.
