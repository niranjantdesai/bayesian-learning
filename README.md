# Bayesian Learing

## Setup
* Use the environment.yml file to setup a conda environment

* Install the PBP code using the instructions provided and copy the .so file generated to the conda env libs folder

## Acknowledgments

- Monte Carlo droput: Yarin Gal's [demo](https://colab.research.google.com/drive/1zcOYplMmun83cL59G1VA4G8HuJAU_neF) from MLSS Moscow 2019
- Bayes by Backprop: https://github.com/ThirstyScholar/bayes-by-backprop
- SGLD code: 
- For one experiment, we started from this [notebook](https://colab.research.google.com/drive/1vV5bsp7o6SyhAXErHwUC1FYxb-9Dc9SK) but modified it to use a third party SGLD optimizer obtained from [PySGMCMC ](https://pysgmcmc.readthedocs.io/en/pytorch/index.html).
- For the second experiment: https://github.com/fregu856/evaluating_bdl/blob/master/toyRegression/SGLD-64/train.py Modified it and visualized after tuning it for a toy example.
- PBP: We used the author's code as-is, provided on [Github](https://github.com/HIPS/Probabilistic-Backpropagation).



# Running instructions

* To run SGLD experiment: python code/sgld_final_code/SGLD-64/train.py
* Jupyter notebooks for SGLD and PBP can be run from the code directory as the base
* Jupyter notebooks for Monte Carlo dropout and Bayes by Backprop can be run in Google Colab directly
