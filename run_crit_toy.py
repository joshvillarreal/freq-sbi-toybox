import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from keras import metrics
import joblib
from sim import *
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
model = load_model('models/model.keras')
pre_sigmoid_model= load_model("models/pre_sig_model.keras", compile=False)
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=1e-4)
pre_sigmoid_model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=[metrics.AUC()])
scaler = joblib.load("models/scaler.joblib")

GRIDSIZE = 30
LOG_SIN2_2THETA_BOUNDS = (-4., 0.)
LOG_DELTA_M2_BOUNDS = (0., 3.)
log_sin2_2theta_grid = np.linspace(*LOG_SIN2_2THETA_BOUNDS, GRIDSIZE)
log_delta_m2_grid = np.linspace(*LOG_DELTA_M2_BOUNDS, GRIDSIZE)
THETAS = np.array([[logsinsq, logdmsq] for logsinsq in log_sin2_2theta_grid for logdmsq in log_delta_m2_grid])

def compute_dllh_estimate(theta, x, pre_sigmoid_model, scaler, M, posterior_gridsize):
    """
    Estimate of dLLH between gridpoint theta and best fit point (MAP) for experimental realization x

    theta : array-like
        Grid-point at which to compute test statistic
    x : array-like
        Raw experimental realization
    pre_sigmoid_model
        Trained subnetwork consisting of all layers except final activation
    scaler
        MinMaxScaler used for input data preparation
    M : int
        Number of samples to use in MC mean
    posterior_gridsize : int
        Gridsize to use for posterior computation (and thus granularity of MAP bfp estimate)
    """
    # getting the MAP
    log_sin2_2theta_grid = np.linspace(*LOG_SIN2_2THETA_BOUNDS, posterior_gridsize)
    log_delta_m2_grid = np.linspace(*LOG_DELTA_M2_BOUNDS, posterior_gridsize)
    thetas_for_posterior = np.array(
        [[logsinsq, logdmsq] for logsinsq in log_sin2_2theta_grid for logdmsq in log_delta_m2_grid]
    )
    log_posterior = compute_log_posterior(thetas_for_posterior, x, M, pre_sigmoid_model, scaler)
    bfp = thetas_for_posterior[np.argmax(log_posterior)]

    # computing likelihood ratio estimate
    inp_scaled = _make_network_input(x.reshape(1, -1), theta.reshape(1, -1), bfp.reshape(1, -1), scaler)
    return pre_sigmoid_model.predict(inp_scaled, verbose=0).flatten()[0], bfp

def _fill_neg_inf_with_min(arr):
    """Fills negative infinities in a NumPy array with its smallest value.

    Args:
        arr (np.ndarray): The input NumPy array.

    Returns:
        np.ndarray: A new array with negative infinities replaced.
    """
    arr = np.array(arr, dtype=float)
    b = np.sort(arr)
    min_val = b[~np.isneginf(b)][1]
    arr[np.isneginf(arr)] = min_val
    return arr


def _make_network_input(inpx, inpth, inpthp, scaler):
    inp = np.hstack((inpx, inpth, inpthp))
    inp_scaled = scaler.transform(inp)
    return inp_scaled


def compute_log_posterior(thetas, x, M, pre_sigmoid_model, scaler):
    """
    thetas : np.ndarray
        List of grid points in model parameter space on which to evaluate posterior density
    x : array-like
        Raw experimental realization
    M : int
        Number of samples to use in MC mean
    pre_sigmoid_model
        Trained subnetwork consisting of all layers except final activation
    scaler
        MinMaxScaler used for input data preparation
    """
    N = thetas.shape[0]

    # using flat prior
    log_prior = np.log(1 / (
        (LOG_SIN2_2THETA_BOUNDS[1] - LOG_SIN2_2THETA_BOUNDS[0]) * \
        (LOG_DELTA_M2_BOUNDS[1] - LOG_DELTA_M2_BOUNDS[0])
    ))
    log_M = np.log(M)

    # sample prior M times, shared across all thetas
    theta_ps = generate_from_uniform_prior(M)  # shape: (M, prior_dim)

    # prepare nn inputs
    inpx = np.repeat(np.array([x]), N * M, axis=0)  # shape: (N*M, x_dim)
    inpth = np.repeat(thetas, M, axis=0)            # shape: (N*M, 2)
    theta_ps_tiled = np.tile(theta_ps, (N, 1))      # shape: (N*M, prior_dim)

    inp_scaled = _make_network_input(inpx, inpth, theta_ps_tiled, scaler) # shape: (N*M, total_input_dim)

    # predict
    pre_sigmoid_preds = pre_sigmoid_model.predict_on_batch(inp_scaled)  # shape: (N*M, 1) or (N*M,)
    pre_sigmoid_preds = pre_sigmoid_preds.reshape(N, M)

    # logpost
    log_posteriors = log_prior + log_M - np.log(np.exp(-pre_sigmoid_preds).sum(axis=1))

    return _fill_neg_inf_with_min(np.array(log_posteriors))

def generate_from_uniform_prior(n_samples):
    log_sin2_2theta_samples = np.random.uniform(*LOG_SIN2_2THETA_BOUNDS, n_samples)
    log_delta_m2_samples = np.random.uniform(*LOG_DELTA_M2_BOUNDS, n_samples)
    return np.column_stack((log_sin2_2theta_samples, log_delta_m2_samples))

## COMPUTING CRITICAL VALUES
ALPHAS = np.array([0.5,0.6,0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99])
K = 100
neg2dllh_estimate_crit_dict = {
    "sin2_2theta": [],
    "delta_m2": [],
}
for alpha in ALPHAS:
    neg2dllh_estimate_crit_dict[f"-2dllh_estimate_crit_{alpha}"] = []

M=300
for th in tqdm(THETAS):
    # simulate K experiments and compute dchi2s
    dllh_estimates_temp = []
    log10_sin2_2theta = th[0]
    log10_delta_m2 = th[1]
    for _ in range(K):
        x_temp, _ = simulate_counts(
            sin2_2theta=10.**log10_sin2_2theta,
            delta_m2=10.**log10_delta_m2
        )

        dllh_estimate_temp, bfp = compute_dllh_estimate(
            theta=th,
            x=x_temp,
            pre_sigmoid_model=pre_sigmoid_model,
            scaler=scaler,
            M=M,
            posterior_gridsize=GRIDSIZE
        )
        dllh_estimates_temp.append(-2*dllh_estimate_temp.item())
    # compute alpha percentile and append
    for ALPHA in ALPHAS:
        dllh_estimate_crit = np.quantile(dllh_estimates_temp, ALPHA) # this sign matters!
        neg2dllh_estimate_crit_dict[f"-2dllh_estimate_crit_{ALPHA}"].append(dllh_estimate_crit)

    neg2dllh_estimate_crit_dict["sin2_2theta"].append(10.**th[0])
    neg2dllh_estimate_crit_dict["delta_m2"].append(10.**th[1])

neg2dllh_estimate_crit_df = pd.DataFrame(neg2dllh_estimate_crit_dict)
neg2dllh_estimate_crit_df.to_csv("output/neg2dllh_estimate_crit_df_full.csv")


## COMPUTING COVERAGE NOW

num_samples=50
row_indices = np.random.choice(THETAS.shape[0], size=25, replace=False)
sample_params = THETAS[row_indices]


cov = {
    "sin2_2theta": [],
    "delta_m2": [],
}
for alpha in ALPHAS:
    cov[f"coverage_{alpha}"] = []


for th in sample_params:
    print(10**th)
    coverage_counts = {alpha: 0 for alpha in ALPHAS}
    for i in tqdm(range(0,num_samples)):
        counts, _ = simulate_counts(
        sin2_2theta=10**th[0],
        delta_m2=10**th[1]
        )
        dllh, bfp = compute_dllh_estimate(
        theta=th,
        x=counts,
        pre_sigmoid_model=pre_sigmoid_model,
        scaler=scaler,
        M=M,
        posterior_gridsize=GRIDSIZE
        )
        neg_2_dllh = -2*dllh
        neg_2llh_df_filtered = neg2dllh_estimate_crit_df.loc[(np.round(neg2dllh_estimate_crit_df['sin2_2theta'], 5) == np.round(10**th[0], 5)) & (np.round(neg2dllh_estimate_crit_df['delta_m2'], 5) == np.round(10**th[1], 5))]
        print(neg_2llh_df_filtered)
        for alpha in ALPHAS:
            if neg_2_dllh< neg_2llh_df_filtered[f'-2dllh_estimate_crit_{alpha}'].iloc[0]:
                coverage_counts[alpha] +=1
    cov['sin2_2theta'].append(th[0])
    cov['delta_m2'].append(th[1])
    for alpha in ALPHAS:
        cov[f"coverage_{alpha}"].append(
            coverage_counts[alpha] / num_samples
        )
        print(coverage_counts[alpha])

cov = pd.DataFrame(cov)
cov.to_csv("output/coverages.csv", index=False)
