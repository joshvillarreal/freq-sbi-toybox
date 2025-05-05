import numpy as np
from scipy.integrate import quad


class OscillationSimulator:
    """
    Simulates electron neutrino appearance in a toy oscillation model.

    Attributes:
        Lmin (float): Minimum baseline in km.
        Lmax (float): Maximum baseline in km.
        flux_norm (float): Normalization constant for expected flux at reference probability.
        reference_prob (float): Reference oscillation probability.
        background (float): Expected background count per bin.
    """

    def __init__(self, Lmin=0.6, Lmax=1.0, flux_norm=100, reference_prob=0.01, background=100):
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.flux_norm = flux_norm
        self.reference_prob = reference_prob
        self.background = background

    def prob_nue_app(self, E, L, sin22th, dm2):
        """Oscillation probability for \(\nu_\mu \rightarrow \nu_e\)"""
        return sin22th * np.sin(1.27 * dm2 * L / E) ** 2

    def prob_nue_app_Eavg(self, Emin, Emax, L, sin22th, dm2):
        """Energy-averaged oscillation probability at fixed L"""
        integrand = lambda E: self.prob_nue_app(E, L, sin22th, dm2) / (Emax - Emin)
        return quad(integrand, Emin, Emax)[0]

    def expected_osc_counts(self, Emin, Emax, sin22th, dm2):
        """Expected number of oscillated signal events in an energy bin"""

        def integrand(L):
            p_avg = self.prob_nue_app_Eavg(Emin, Emax, L, sin22th, dm2)
            flux = (self.flux_norm / self.reference_prob)
            return p_avg * flux / (self.Lmax - self.Lmin)

        return quad(integrand, self.Lmin, self.Lmax)[0]

    def simulate(self, N_samples, sin22th, dm2, Ebins=None):
        """
        Simulate observed event counts for each energy bin.

        Args:
            N_samples (int): Number of Monte Carlo samples to generate.
            sin22th (float): Oscillation amplitude.
            dm2 (float): Mass-squared difference in eV^2.
            Ebins (list of tuples): List of (Emin, Emax) bin edges in GeV. Defaults to 10-GeV bins from 10 to 60.

        Returns:
            list of np.ndarray: Simulated observed event counts, one array per energy bin.
        """
        if Ebins is None:
            Ebins = [(Emin, Emin + 10) for Emin in np.arange(10., 60., 10.)]

        all_counts = []

        for Emin, Emax in Ebins:
            signal = self.expected_osc_counts(Emin, Emax, sin22th, dm2)
            total_expected = self.background + signal
            observed = np.random.poisson(total_expected, size=N_samples)
            all_counts.append(observed)

        return all_counts
