PoissonQN2_MultiMixture.py
 - contains code for running QN2 and EM algorithms

PoissonQN2_Multi_TestRuns.py
 - contains code for running test problems using algorithms QN2 and EM in PoissonQN2_MultiMixture.py
 - user needs to specify NUM_MIXTURES and NUM_RUNS
 - results must be pickled with name structure specified in Analysis.py (see Data Files for example)

Analysis.py
 - Takes results and generates plots.  Data files must be present in folder and be in .pk1 format
 - User needs only to press go (as long as data filenames consistent)

Data Files
 - RunData2_K2_init3.pk1   (2 mixtures, 3 initializations)
 - RunData2_K5_init3.pk1   (5 mixtures, 3 initializations)
 - RunData2_K10_init3.pk1  (10 mixtures, 3 initializations)

utils.py
 - contains ancillary functions used by all other scripts
