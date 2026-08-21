# freq-sbi-toybox

[Julia Woodward](mailto:julia785@mit.edu)

This branch of the repository provides code to perform [direct amortized neural likelihood ratio estimation](https://arxiv.org/abs/2311.10571) on a toy neutrino oscillation experiment; the same presented by [Feldman and Cousins](https://arxiv.org/abs/physics/9711021), as well as evaluate parameter goodness-of-fit (PG) tension between two experimental results using direct amortized neural likelihood ratio estimation. 

The files in this repository serve the following purposes:

1. **`sim.py`**  — Contains functions for generating pseudo-experiments from the toy neutrino oscillation simulation. These functions are used to simulate experimental realizations at different oscillation parameter points and provide the training and validation data used throughout the analysis.
2. **`toybox_training.ipynb`** — Trains neural networks to estimate likelihood ratios using DNRE. This notebook trains networks for the individual toy experiments as well as a network for their combined dataset, allowing both single-experiment and joint inference to be performed.
3. **`toy_functions.ipynb`**  — Contains the fitting and inference functions used throughout the toy analysis. These functions evaluate neural likelihood-ratio estimates, perform fits over the oscillation parameter space, and compute the corresponding exact likelihood-ratio or delta_chi2 values for comparison.
4. **`toybox.ipynb`** — Performs fits to pseudo-experimental realizations using both the SBI-evaluated likelihood ratio and the exact likelihood ratio. The results from the two approaches are compared, and the coverage properties of confidence regions constructed using the SBI-based test statistic are evaluated.
5. **`toy_tension.ipynb`** — Evaluates parameter goodness-of-fit (PG) tension between two toy experimental results. The PG test statistic is computed using both the exact likelihood calculation and the SBI-approximated likelihood-ratio estimates, allowing the agreement between the two methods to be studied empirically.

