import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    x = np.array(x)
    return lam * np.exp(-lam * x) * (x >= 0)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """
    P_A = 0.3
    P_B = 0.7

    mu_A, sigma_A = 40, np.sqrt(2)
    mu_B, sigma_B = 45, np.sqrt(2)

    likelihood_A = gaussian_pdf(time, mu_A, sigma_A)
    likelihood_B = gaussian_pdf(time, mu_B, sigma_B)

    numerator = likelihood_B * P_B
    denominator = likelihood_A * P_A + likelihood_B * P_B

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    P_A = 0.3
    P_B = 0.7

    mu_A, sigma_A = 40, 2
    mu_B, sigma_B = 45, 2

    # generate group labels
    groups = np.random.choice(['A', 'B'], size=n, p=[P_A, P_B])

    times = np.zeros(n)

    # generate finishing times
    times[groups == 'A'] = np.random.normal(mu_A, sigma_A, np.sum(groups == 'A'))
    times[groups == 'B'] = np.random.normal(mu_B, sigma_B, np.sum(groups == 'B'))

    # small window around the observed time
    epsilon = 0.5
    mask = np.abs(times - time) < epsilon

    if np.sum(mask) == 0:
        return 0

    return np.mean(groups[mask] == 'B')
