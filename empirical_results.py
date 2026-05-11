import svvamp
import numpy as np

# Parameters
m_values = range(3, 11)       
n_values = [100, 1000, 10000]    
num_profiles = 10000

results = {}

for m in m_values:
    results[m] = {}
    for n in n_values:

        # Function to generate the profiles with m candidates 
        # and n electors
        generator = svvamp.GeneratorProfileIc(n_v=n, n_c=m)

        count_cw_not_irv = 0

        for i in range(num_profiles):

            # Generate profile
            profile = generator()

            # Find the IRV winner of the profile
            irv = svvamp.RuleIRV()(profile)
            irv_winner = irv.w_

            # Find the CW (if there is one)
            cw = profile.condorcet_winner_rk

            # If there is a CW which is not the IRV 
            # winner, add 1 to the count
            if not np.isnan(cw) and cw != irv_winner:
                count_cw_not_irv += 1

        p_hat = count_cw_not_irv / num_profiles
        results[m][n] = p_hat
        print(f"With m={m} candidates and n={n} electors, the empirical probability is {p_hat:.3f}")

# ===== PRINT TABLE =====
print("\n" + "=" * 45)
print("P(There is a CW that IRV fails to elect)")
print(f"({num_profiles} profiles, Impartial Culture)")
print("=" * 45)

header = f"{'m':>4} | " + " | ".join(f"{'n='+str(n):^8}" for n in n_values)
print(header)
print("-" * len(header))

for m in m_values:
    row = f"{m:>4} | " + " | ".join(f"{results[m][n]:^8.3f}" for n in n_values)
    print(row)

print("=" * 45)