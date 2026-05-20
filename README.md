# Probability Calculator: Condorcet Winner Not Winning IRV

This repository computes the probability that there exists a **Condorcet Winner (CW) that is not elected by Instant-Runoff Voting (IRV)**, under the **Impartial Culture (IC)** assumption. It provides both a theoretical calculator (exact asymptotic probability as the number of voters tends to infinity) and an empirical simulator (Monte Carlo estimation for finite electorates).

---

## Background

**Instant-Runoff Voting (IRV)** is a voting system in which the candidate with fewest first-preference votes is eliminated at each round until one candidate holds a majority.

A **Condorcet Winner** is a candidate who defeats every other candidate in pairwise majority comparisons. When a CW exists, many argue they are the most democratically legitimate winner. However, IRV does not always elect the CW when one exists.

This project quantifies the probability of this failure under the **Impartial Culture** assumption, in which each voter's preference ranking is drawn uniformly and independently at random from all $m!$ possible orderings.

---

## Repository Structure

```
├── theoretical_results.py   # Asymptotic probability via Gaussian orthant integration
├── empirical_results.py     # Monte Carlo simulation for finite n and m
└── requirements.txt         # Python dependencies
```

---

## Theoretical Calculator (`theoretical_results.py`)

Computes the **exact asymptotic probability** $P(\text{There exists a CW and IRV fails to elect it})$ under IC as the number of voters $n \to \infty$, for a given number of candidates $m$.


### Usage

```python
from theoretical_results import compute_irv_cw_probability

# Compute for m = 3 candidates, suppressing covariance matrix output
# If you want to print the covariance matrices,
# set print_cov = True
p = compute_irv_cw_probability(m=3, print_cov=False)
```

Running the script directly computes results for $m = 3, 4, 5$:

```bash
python theoretical_results.py
```

Sample output:
```
==================================================
  m = 3 candidates
==================================================
  k= 1 | d=   3 | P(A*_k)=0.034000 | weight=1 | contribution=0.034000

  Total probability = 0.034
```

---

## Condorcet Winner Existence Theoretical Calculator (`CW_theoretical_results.py`)

Computes the **asymptotic probability** $P(\text{There exists a CW})$ under IC as the number of voters $n \to \infty$, for a given number of candidates $m$.


### Usage

```python
from CW_theoretical_results import condorcet_winner_probability

# Compute for m = 3 candidates
p = condorcet_winner_probability(m=3)
```

Running the script directly computes results for $m = 3$ to $m=20$:

```bash
python theoretical_results.py
```

Sample output:
```
m = 3
P(specific candidate is CW) = 0.304087
P(any CW exists)            = 0.912260
```

---

## Empirical Simulator (`empirical_results.py`)

Estimates the probability via **Monte Carlo simulation** using the `svvamp` library to generate random preference profiles and compute IRV and Condorcet winners.

### What it computes

For each combination of candidates $m = \{3, \ldots, 10\}$ and voters $n$, it draws `num_profiles` independent IC preference profiles and estimates:

$$\hat{p} = \frac{\text{profiles where CW exists and IRV elects someone else}}{\text{total profiles}}$$

### Usage

Edit the parameters at the top of the script:

```python
m_values     = range(3, 11)       # number of candidates
n_values     = [10, 100, 1000]    # number of voters per profile
num_profiles = 5000               # simulations per (m, n) cell
```

Then run:

```bash
python empirical_results.py
```

Output is a formatted table, e.g.:

```
=============================================
P(There is a CW that IRV fails to elect) 
(10000 profiles, Impartial Culture)
=============================================
   m |  n=100    |  n=1000   |  n=10000 
-----+----------+----------+---------
   3 |  0.023  |  0.029  |  0.034
   4 |  0.038  |  0.049  |  0.057
  ...
```

## Installation

Python 3.8+ is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---------|---------|
| numpy   | 2.4.4   |
| scipy   | 1.17.1  |
| svvamp  | 0.13.0  |

---

## Interpreting the Results

The theoretical values represent the limiting probability as $n \to \infty$. The empirical values converge toward the theoretical ones as both $n$ and the `num_profiles` T grow. 

| m  | Theoretical (asymptotic) | Empirical (n=10,000, T=10,000) |
|----|--------------------------|--------------------------------|
| 3  | ~0.034                   | 0.034                          |
| 4  | ~0.059                   | 0.057                          |
| 5  | ~0.076                   | 0.075                          |
| 6  | ~0.087                   | 0.087                          |
| 7  | ~0.094                   | 0.093                          |
| 8  | ~0.099                   | 0.099                          |
| 9  | ~0.102                   | 0.099                          |
| 10 | ~0.104                   | 0.103                          |

The empirical estimates (10,000 voters, 10,000 profiles) are in close agreement with the theoretical asymptotic values, with a maximum deviation of 0.003, consistent with the expected Monte Carlo sampling error at this profile count.

---
