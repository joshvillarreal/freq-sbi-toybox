import numpy as np
from sim import *

GRIDSIZE = 30
LOG_SIN2_2THETA_BOUNDS = (-4., 0.)
LOG_DELTA_M2_BOUNDS = (0., 3.)
log_sin2_2theta_grid = np.linspace(*LOG_SIN2_2THETA_BOUNDS, GRIDSIZE)
log_delta_m2_grid = np.linspace(*LOG_DELTA_M2_BOUNDS, GRIDSIZE)
THETAS = np.array([[logsinsq, logdmsq] for logsinsq in log_sin2_2theta_grid for logdmsq in log_delta_m2_grid])


def compute_dllh_estimate_grid(x, pre_sigmoid_model, scaler, M, posterior_gridsize):
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
    N = len(THETAS)
    #print(N)
    inp_scaled = _make_network_input(np.array([x]*N), THETAS, np.array([bfp]*N), scaler)
    #print(inp_scaled)
    return pre_sigmoid_model.predict(inp_scaled, verbose=0).flatten()

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

def compute_delta_chi2(x, sin2_2theta, delta_m2):
    out = 0.
    for i, E in enumerate(ENERGY_CENTERS):
        # getting mu_i, expected oscillation contributions
        P_avg = average_probability_over_L(sin2_2theta, delta_m2, E)
        mu_i = NORMALIZATION * P_avg
        
        # getting mu_best_i
        mu_best_i = x[i] - BACKGROUND_PER_BIN
        
        # update sum
        out += 2 * (mu_i - mu_best_i + x[i]*np.log((mu_best_i+BACKGROUND_PER_BIN)/(mu_i+BACKGROUND_PER_BIN)))
    return out

