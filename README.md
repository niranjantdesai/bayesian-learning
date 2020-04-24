# Bayesian Learing

## Epistemic Uncertainity

Methods planned:
1. PBP (code given )
2. SGLD (code given)
3. Bayesian Dark Knowledge (tbd)

Experiments planned:
1. Number of layers
2. Training dataset size
3. Dataset distribution

## Setup
* Use the environment.yml file to setup a conda environment

* Install the PBP code using the instructions provided and copy the .so file generated to the conda env libs folder



## Acknowledgments

- SGLD code: we started from this notebook https://colab.research.google.com/drive/1vV5bsp7o6SyhAXErHwUC1FYxb-9Dc9SK but modified it to use a third party SGLD optimizer obtained from [PySGMCMC ](https://pysgmcmc.readthedocs.io/en/pytorch/index.html).
- PBP: We used the author's code as-is, provided on [Github](https://github.com/HIPS/Probabilistic-Backpropagation).