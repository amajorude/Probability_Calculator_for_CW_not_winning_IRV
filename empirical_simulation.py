import svvamp
import numpy as np

# Parameters
n_voters = 10000   # voters per profile
m = 4           # number of candidates
num_profiles = 5000

# Generator (Impartial Culture)
generator = svvamp.GeneratorProfileIc(n_v=n_voters, n_c=m)

count_cw_not_irv = 0
count_cw_exists = 0

for i in range(num_profiles):
    profile = generator()

    # IRV winner
    irv = svvamp.RuleIRV()(profile)
    irv_winner = irv.w_

    # Condorcet winner
    cw = profile.condorcet_winner_rk
    if not np.isnan(cw):
        count_cw_exists += 1

        if cw != irv_winner:
            count_cw_not_irv += 1

    # Optional progress display
    if (i+1) % 1000 == 0:
        print(f"{i+1} profiles processed...")

# Results

p_hat = count_cw_not_irv / num_profiles

# Standard error
se = np.sqrt(p_hat * (1 - p_hat) / num_profiles)

# 95% confidence interval
ci_lower = p_hat - 1.96 * se
ci_upper = p_hat + 1.96 * se

print("\n===== RESULTS =====")
print("Total profiles:", num_profiles)
print("Profiles with CW:", count_cw_exists)
print("CW not elected by IRV:", count_cw_not_irv)

if count_cw_exists > 0:
    print("Conditional probability P(IRV fails | CW exists):",
          count_cw_not_irv / count_cw_exists)

print(f"Unconditional probability: {p_hat:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")