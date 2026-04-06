import numpy as np
from math import comb, factorial, prod
from scipy.stats import multivariate_normal
import warnings

warnings.filterwarnings("ignore")


def mvncdf_lower_upper(lower, upper, mean, cov):
    """
    Compute P(lower < X < upper) for a multivariate normal X ~ N(mean, cov)
    using scipy's built-in multivariate normal CDF (Genz algorithm).
    """
    # P(lower < X < upper) = CDF(upper) - ... handled via lower_limit parameter
    prob = multivariate_normal.cdf(
        upper, mean=mean, cov=cov, lower_limit=lower, allow_singular=True
    )
    return float(prob)


def main():
    m = 4  # number of candidates
    m_fact = factorial(m)

    def sum_expr_closed(l, m):
        part1 = sum(
            comb(l - 1, s) * factorial(s + 1) * factorial(m - 2 - s)
            for s in range(l)
        )
        part2 = 0.5 * sum(
            comb(l - 1, s) * factorial(s) * factorial(m - 1 - s)
            for s in range(l)
        )
        return part1 + part2

    def falling(x, n):
        """Falling factorial: x * (x-1) * ... * (x-n+1)"""
        if n == 0:
            return 1
        return prod(range(x, x - n, -1))

    def c_l(l, m):
        return 1.0 / (m - l)

    final_prob = 0.0
    print(f"\n===== m = {m} =====")

    for k in range(1, m - 1):
        d = int((k + 1) * (m - k / 2) - 1)
        d1 = k * m - k * (k + 1) // 2

        Sigma = np.zeros((d, d))

        print(f"\n===== k = {k} =====")
        print(f"d = {d}")
        print(f"d1 = {d1}")

        # Fill upper-left block
        for l in range(k):
            i = l * m - l * (l + 1) // 2  # 0-based starting index for row block
            for s in range(1, m - l - 1 + 1):
                i += 1
                i_idx = i - 1  # convert to 0-based

                for l2 in range(k):
                    j = l2 * m - l2 * (l2 + 1) // 2
                    for t in range(1, m - l2 - 1 + 1):
                        j += 1
                        j_idx = j - 1  # convert to 0-based

                        # DIAGONAL
                        if i == j:
                            Sigma[i_idx, i_idx] = 2 * c_l(l, m)

                        # AGAINST ROUND L2
                        elif j <= d1:
                            # AGAINST SAME ROUND
                            if l2 == l:
                                Sigma[i_idx, j_idx] = c_l(l, m)

                            # AGAINST DIFFERENT ROUND
                            else:
                                a = l + s + 1
                                if l2 < k - 1:
                                    c = l2 + t + 1
                                    d_aux = l2 + 1
                                else:
                                    c = k - 1 + t
                                    d_aux = m

                                if a == d_aux:
                                    Sigma[i_idx, j_idx] = -c_l(l, m)
                                elif a == c:
                                    Sigma[i_idx, j_idx] = c_l(l, m)

                # AGAINST CW CONDITION
                for t in range(1, m):
                    col = d1 + t  # 1-based column
                    col_idx = col - 1  # 0-based

                    # LAST ROUND
                    if l == k - 1:
                        a = k - 1 + s
                        if a == t:
                            Sigma[i_idx, col_idx] = -c_l(l, m)
                        elif t >= l + 1:
                            Sigma[i_idx, col_idx] = -c_l(l, m) / 2
                        else:  # t < l + 1
                            Sigma[i_idx, col_idx] = (
                                -sum_expr_closed(l, m) / m_fact + c_l(k - 2, m) / 2
                            )

                    # NOT LAST ROUND
                    else:
                        a = l + 1 + s
                        b = l + 1
                        if a == m:
                            if t == l + 1:
                                Sigma[i_idx, col_idx] = c_l(l, m)
                            elif t > l + 1:
                                Sigma[i_idx, col_idx] = c_l(l, m) / 2
                            else:  # t < l + 1
                                Sigma[i_idx, col_idx] = (
                                    sum_expr_closed(l, m) / m_fact - c_l(l - 1, m)
                                )
                        elif a == t:
                            Sigma[i_idx, col_idx] = -c_l(l, m) / 2
                        elif b == t:
                            Sigma[i_idx, col_idx] = c_l(l, m) / 2

        # Final block (CW * CW)
        for i in range(1, m):
            for j in range(1, m):
                row_idx = d1 + i - 1  # 0-based
                col_idx = d1 + j - 1  # 0-based
                if i == j:
                    Sigma[row_idx, col_idx] = 1 / 4
                else:
                    Sigma[row_idx, col_idx] = 1 / 12

        # Symmetrize
        Sigma = np.triu(Sigma) + np.triu(Sigma, 1).T

        muY = np.zeros(d)
        lower = np.zeros(d)
        upper = np.full(d, np.inf)

        prob = mvncdf_lower_upper(lower, upper, muY, Sigma)

        print(f"Probability in round {k} = {prob:.4f}")

        final_prob += falling(m - 1, k - 1) * prob

    print("\n===== Total Probability =====")
    print(f"Total Probability = {m * final_prob:.4f}")


if __name__ == "__main__":
    main()

