import numpy as np
from math import prod
import warnings

warnings.filterwarnings("ignore")


def mvn_prob_mc(mean, cov, n_samples=20000, batch_size=2000):
    """
    Monte Carlo estimate of P(X > 0) for X ~ N(mean, cov)
    """
    d = len(mean)
    count = 0
    total = 0

    for _ in range(n_samples // batch_size):
        samples = np.random.multivariate_normal(mean, cov, size=batch_size)
        count += np.sum(np.all(samples > 0, axis=1))
        total += batch_size

    return count / total


def main():
    m = 3  # number of candidates

    def falling(x, n):
        if n == 0:
            return 1
        return prod(range(x, x - n, -1))

    # Precompute c_l values (small speedup)
    c_vals = [1.0 / (m - l) for l in range(m)]

    final_prob = 0.0
    print(f"\n===== m = {m} =====")

    for k in range(1, m - 1):
        d = int((k + 1) * (m - k / 2) - 1)
        d1 = k * m - k * (k + 1) // 2

        Sigma = np.zeros((d, d))

        print(f"\n===== k = {k} =====")
        print(f"d = {d}")

        # ---- SAME MATRIX CONSTRUCTION (unchanged) ----
        for l in range(k):
            i = l * m - l * (l + 1) // 2
            for s in range(1, m - l):
                i += 1
                i_idx = i - 1

                for l2 in range(k):
                    j = l2 * m - l2 * (l2 + 1) // 2
                    for t in range(1, m - l2):
                        j += 1
                        j_idx = j - 1

                        if i == j:
                            Sigma[i_idx, i_idx] = 2 * c_vals[l]

                        elif j <= d1:
                            if l2 == l:
                                Sigma[i_idx, j_idx] = c_vals[l]
                            else:
                                a = l + s + 1
                                if l2 < k - 1:
                                    c = l2 + t + 1
                                    d_aux = l2 + 1
                                else:
                                    c = k - 1 + t
                                    d_aux = m

                                if a == d_aux:
                                    Sigma[i_idx, j_idx] = -c_vals[l]
                                elif a == c:
                                    Sigma[i_idx, j_idx] = c_vals[l]

                for t in range(1, m):
                    col = d1 + t
                    col_idx = col - 1

                    if l == k - 1:
                        a = k - 1 + s
                        if a == t:
                            Sigma[i_idx, col_idx] = -c_vals[l]
                        elif t >= l + 1:
                            Sigma[i_idx, col_idx] = -c_vals[l] / 2
                        else:
                            Sigma[i_idx, col_idx] = -c_vals[k - 2] / 2
                    else:
                        a = l + 1 + s
                        b = l + 1
                        if a == m:
                            if t == l + 1:
                                Sigma[i_idx, col_idx] = c_vals[l]
                            elif t > l + 1:
                                Sigma[i_idx, col_idx] = c_vals[l] / 2
                            else:
                                Sigma[i_idx, col_idx] = c_vals[l - 1] / 2
                        elif a == t:
                            Sigma[i_idx, col_idx] = -c_vals[l] / 2
                        elif b == t:
                            Sigma[i_idx, col_idx] = c_vals[l] / 2

        for i in range(1, m):
            for j in range(1, m):
                row_idx = d1 + i - 1
                col_idx = d1 + j - 1
                if i == j:
                    Sigma[row_idx, col_idx] = 1 / 4
                else:
                    Sigma[row_idx, col_idx] = 1 / 12

        Sigma = np.triu(Sigma) + np.triu(Sigma, 1).T

        # ---- Monte Carlo instead of CDF ----
        muY = np.zeros(d)

        prob = mvn_prob_mc(muY, Sigma, n_samples=20000)

        print(f"Probability in round {k} ≈ {prob:.4f}")

        final_prob += falling(m - 1, k - 1) * prob

    print("\n===== Total Probability =====")
    print(f"Total Probability ≈ {m * final_prob:.4f}")


if __name__ == "__main__":
    main()