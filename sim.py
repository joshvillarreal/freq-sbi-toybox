import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

NUM_BINS = 5
ENERGY_BINS = np.linspace(10, 60, NUM_BINS + 1) # GeV
ENERGY_CENTERS = 0.5 * (ENERGY_BINS[:-1] + ENERGY_BINS[1:])
BASELINE_RANGE = (0.600, 0.1000) # km
BACKGROUND_PER_BIN = 100
NORMALIZATION = 100 / 0.01  # So that P=0.01 => 100 signal events


def oscillation_probability(sin2_2theta, delta_m2, L, E):
    """Two-flavor oscillation probability: P(νμ → νe)"""
    return sin2_2theta * np.sin(1.267 * delta_m2 * L / E) ** 2


def average_probability_over_L(sin2_2theta, delta_m2, E, num_samples=1000):
    """Average the probability over uniformly distributed baseline L in [600, 1000] m"""    
    integrand = lambda l: oscillation_probability(sin2_2theta, delta_m2, l, E)
    return quad(integrand, *BASELINE_RANGE)[0] / (BASELINE_RANGE[1] - BASELINE_RANGE[0])


def simulate_counts(sin2_2theta, delta_m2, seed=None):
    """Simulate total observed counts per energy bin"""
    rng = np.random.default_rng(seed)
    signal = []
    total_counts = []
    for E in ENERGY_CENTERS:
        P_avg = average_probability_over_L(sin2_2theta, delta_m2, E)
        expected_signal = NORMALIZATION * P_avg
        total_expected = BACKGROUND_PER_BIN + expected_signal
        if total_expected < 0:
            print(sin2_2theta, delta_m2, total_expected)
        total_counts.append(rng.poisson(total_expected))
        signal.append(expected_signal)
    return np.array(total_counts), np.array(signal)


def _build_title(sin2_2theta, delta_m2):
    title = r"$\sin^2 2 \theta = $"
    title += "%.3f" % sin2_2theta
    title += r", $\Delta m^2 = $"
    title += "%.3f" % delta_m2
    title += r" eV$^2$"
    return title


def plot_event_display(sin2_2theta, delta_m2, counts, expected_signal):
    """Plot observed counts with oscillation probability overlay"""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Simulated counts
    ax1.stairs(counts, edges=ENERGY_BINS, label='Observed Counts')
    ax1.set_xlabel("Neutrino Energy (GeV)")
    ax1.set_ylabel("Counts")
    ax1.set_title(_build_title(sin2_2theta, delta_m2))
    
    # Expected counts
    expected_counts = expected_signal+BACKGROUND_PER_BIN
    ax1.errorbar(
        ENERGY_CENTERS,
        expected_counts,
        yerr=np.sqrt(expected_counts),
        label='Expected counts',
        color='skyblue'
    )
    
    # Twin axis for oscillation probability
    ax2 = ax1.twinx()
    probs = [average_probability_over_L(sin2_2theta, delta_m2, E) for E in ENERGY_CENTERS]
    ax2.plot(ENERGY_CENTERS, probs, 'r--o', label=r'$P(\nu_\mu \to \nu_e)$', linewidth=2)
    ax2.set_ylabel("Oscillation Probability", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Combine handles and labels
    all_lines = lines + lines2
    all_labels = labels + labels2

    # Create a single legend
    ax1.legend(all_lines, all_labels, loc="lower left", framealpha=1.)

    plt.tight_layout()
    plt.show()
    
