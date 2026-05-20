import numpy as np
from scipy.stats import multivariate_normal


def condorcet_winner_probability(m: int) -> float:
    """
    Probability that a Condorcet winner exists under Impartial Culture
    as the number of voters tends to infinity.
    """
    d = m - 1

    # Covariance matrix:
    # diagonal = 1/4, off-diagonal = 1/12
    Sigma = np.full((d, d), 1 / 12)
    np.fill_diagonal(Sigma, 1 / 4)

    # Orthant probability P(X > 0)
    p_single = multivariate_normal.cdf(
        np.full(d, np.inf),
        mean=np.zeros(d),
        cov=Sigma,
        lower_limit=np.zeros(d),
    )

    # Any of the m candidates can be the CW
    # Same probability by symmetry
    p_total = m * p_single

    print(f"m = {m}")
    print(f"P(candidate m is CW) = {p_single:.3f}")
    print(f"P(any CW exists) = {p_total:.3f}\n")


    return p_total


if __name__ == "__main__":
    for m in range(3, 21):
        condorcet_winner_probability(m)
