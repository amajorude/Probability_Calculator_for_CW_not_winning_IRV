import numpy as np
from math import prod
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def falling_factorial(x: int, n: int) -> int:
    """x * (x-1) * ... * (x-n+1)"""
    return prod(range(x, x - n, -1)) if n > 0 else 1


def c(ell: int, m: int) -> float:
    """Scaling factor for round ell."""
    return 1.0 / (m - ell)


def mvn_orthant_prob(mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute P(X > 0) for X ~ N(mean, cov) via scipy's Genz algorithm.
    """
    d = len(mean)
    lower = np.zeros(d)
    upper = np.full(d, np.inf)
    return float(
        multivariate_normal.cdf(upper, mean=mean, cov=cov,
                                lower_limit=lower, allow_singular=False)
    )


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def elim_block_start(ell: int, m: int) -> int:
    """0-based start of the elimination block for round ell (0-indexed round)."""
    return ell * m - ell * (ell + 1) // 2


def row_elim(ell: int, s: int, m: int) -> int:
    """0-based row index for elimination row (ell, s), s in 1..m-ell-1."""
    return elim_block_start(ell, m) + s - 1


def row_cw(s: int, d1: int) -> int:
    """0-based row index for Condorcet Winner row s, s in 1..m-1."""
    return d1 + s - 1


def d1_size(k: int, m: int) -> int:
    """Number of elimination rows."""
    return k * m - k * (k + 1) // 2


def d_size(k: int, m: int) -> int:
    """Total number of rows."""
    return int((k + 1) * (m - k / 2) - 1)


# ---------------------------------------------------------------------------
# Covariance blocks
# ---------------------------------------------------------------------------

def cov_elim_elim(ell1: int, s1: int, ell2: int, s2: int, k: int, m: int) -> float:
    """
    Cov(Y_{ell1, s1}, Y_{ell2, s2}) for two elimination rows.

    Each elimination row (ell, s) computes:
        n_{S_{ell+1}}(ell + s + 1)  -  n_{S_{ell+1}}(ell + 1)

    where n_{S}(c) = number of voters whose top candidate in S is c,
    scaled so that Var(n_S(c)) = c(ell, m) * (1 - c(ell, m)).

    Covariances arise because the two rows may share candidate sets.
    """
   # a_i is the candidate to keep in row (ell_i, s_i)
   # b_i is the candidate to eliminate in row (ell_i, s_i)
    if ell1 < k - 1:
        a1 = ell1 + s1 + 1
        b1 = ell1 + 1
    else:
        a1 = k - 1 + s1
        b1 = m
    if ell2 < k - 1:
        a2 = ell2 + s2 + 1
        b2 = ell2 + 1
    else:
        a2 = k - 1 + s2
        b2 = m

    # Rows from the same round share the same candidate set S_{ell+1}
    if ell1 == ell2:
        if s1 == s2:
            return 2 * c(ell1, m)         # Diagonal case 
        else:
            return c(ell1, m)              

    # Rows from different rounds: only non-zero if one row's candidate
    # appears in the other row's comparison. Otherwise 0 by symmetry.
    cl = c(min(ell1, ell2), m)

    if a1 == b2:
        return -cl
    if a1 == a2:
        return cl
    if a2 == b1:
        return -cl

    return 0.0


def cov_elim_cw(ell: int, s: int, t: int, k: int, m: int) -> float:
    """
    Cov(Y_{ell, s}, Z_t) where Y_{ell,s} is an elimination row and
    Z_t is the CW row for candidate t (i.e. -(number of votes 
    preferring t over m)).

    The sign structure follows from:
      - Y_{ell,s} has +1 on ballots ranking a = ell+s+1 first in S_{ell+1}
                      -1 on ballots ranking b = ell+1   first in S_{ell+1}
      - Z_t        has -1 on ballots ranking t above m
    """
    is_last = (ell == k - 1)
    cl = c(ell, m)

    if is_last:
        a = k - 1 + s
        if a == t:
            return -cl
        elif t >= ell + 1:
            return -cl / 2
        else:
            return -c(ell - 1, m) / 2 

    else:
        a = ell + 1 + s
        b = ell + 1
        if a == m:
            if t == b:
                return cl
            elif t > b:
                return cl / 2
            else:
                return c(ell - 1, m) / 2 
        elif a == t:
            return -cl / 2
        elif b == t:
            return cl / 2

    return 0.0


def cov_cw_cw(s: int, t: int) -> float:
    """
    Cov(Z_s, Z_t) for two CW rows.
    Z_s = -(votes preferring s over m),  Z_t = -(votes preferring t over m).

    Under IC, each voter's preference between s,m and t,m are independent
    Bernoulli(1/2), giving Var = 1/4 and Cov = 1/12 for distinct pairs.
    """
    return 1 / 4 if s == t else 1 / 12


# ---------------------------------------------------------------------------
# Build the covariance matrix
# ---------------------------------------------------------------------------

def build_covariance(k: int, m: int) -> np.ndarray:
    """Build the full covariance matrix Sigma = A Sigma_z A^T."""
    d  = d_size(k, m)
    d1 = d1_size(k, m)
    Sigma = np.zeros((d, d))

    # -- Elimination vs Elimination block --
    for ell1 in range(k):
        for s1 in range(1, m - ell1):
            i = row_elim(ell1, s1, m)
            for ell2 in range(k):
                for s2 in range(1, m - ell2):
                    j = row_elim(ell2, s2, m)
                    Sigma[i, j] = cov_elim_elim(ell1, s1, ell2, s2, k, m)

    # -- Elimination vs CW block (and its transpose) --
    for ell in range(k):
        for s in range(1, m - ell - 1 + 1):
            i = row_elim(ell, s, m)
            for t in range(1, m):
                j = row_cw(t, d1)
                val = cov_elim_cw(ell, s, t, k, m)
                Sigma[i, j] = val
                Sigma[j, i] = val

    # -- CW vs CW block --
    for s in range(1, m):
        for t in range(1, m):
            i = row_cw(s, d1)
            j = row_cw(t, d1)
            Sigma[i, j] = cov_cw_cw(s, t)

    return Sigma


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_covariance_matrix(Sigma: np.ndarray, k: int, m: int) -> None:
    """Print the covariance matrix with row/column labels."""
    d1 = d1_size(k, m)

    # Build row labels
    labels = []
    for ell in range(k):
        for s in range(1, m - ell):
            a = ell + s + 1
            b = ell + 1
            labels.append(f"E(r={ell+1},+{a},-{b})")
    for t in range(1, m):
        labels.append(f"CW(vs {t})")

    col_width = max(len(l) for l in labels) + 2
    header = " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)

    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(labels):
        row_str = f"{row_label:<{col_width}}"
        for j in range(len(labels)):
            val = Sigma[i, j]
            row_str += f"{val:>{col_width}.4f}"
        print(row_str)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_irv_cw_probability(m: int, verbose: bool = True,
                               print_cov: bool = True) -> float:
    """
    Compute P(candidate m wins IRV and is a Condorcet Winner)
    under the Impartial Culture assumption with m candidates.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"  m = {m} candidates")
        print(f"{'='*50}")

    total = 0.0

    for k in range(1, m - 1):
        Sigma = build_covariance(k, m)

        # Sanity checks
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.any(eigvals < -1e-8):
            print(f"  [WARNING] k={k}: Sigma is not PSD "
                  f"(min eigval={eigvals.min():.2e})")

        if print_cov:
            print(f"\n  --- Covariance matrix for k={k}, m={m} ---\n")
            print_covariance_matrix(Sigma, k, m)

        mean = np.zeros(d_size(k, m))
        prob = mvn_orthant_prob(mean, Sigma)

        weight = falling_factorial(m - 1, k - 1)
        contribution = weight * prob

        if verbose:
            print(f"  k={k:2d} | d={d_size(k,m):4d} | P(A*_k)={prob:.6f} "
                  f"| weight={weight} | contribution={contribution:.6f}")

        total += contribution

    result = m * total
    if verbose:
        print(f"\n  Total probability = {result:.3f}")

    return result


if __name__ == "__main__":
    for m in [4]:
        compute_irv_cw_probability(m, verbose=True, print_cov=False)
