"""
Required Practical 1: "Investigating Stationary Waves".

Code for investigating graphical relation of the
frequency of the first harmonic against length.

Lab report due December 17th, 2025 for Mr. Awuah.
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial

# The data gathered and recorded in my lab book.
# Note that L hasn't been adjusted to be 1/L here yet.
DATA = (
    # (length, frequency)
    (0.3, 250.0),
    (0.4, 200.0),
    (0.5, 166.7),
    (0.6, 125.0),
    (0.7, 111.1),
    (0.8, 83.33),
)

# Adjust L -> 1/L for plotting.
points = tuple((1 / d[0], d[1]) for d in DATA)

# Unpack `POINTS` to x, y arrays.
reciprocal_length, frequency = map(np.array, zip(*points))
print(f"Plotting points: {points}")

# Find the degree 1 (linear) LOBF's coefficients.
poly_lobf = Polynomial.fit(reciprocal_length, frequency, deg=1)
lobf_c, lobf_m = poly_lobf.convert().coef

print(f"LOBF gradient (m): {lobf_m}")
print(f"LOBF y-intercept (c): {lobf_c}")
print(f"LOBF equation: y = {lobf_m:.3f}x + {lobf_c:.3f}")

# Extend the LOBF by `MARGIN` (stylistic choice).
#
# I've done this so the points (marked with crosses)
# don't get cut off when margins are set to zero.
MARGIN = 0.05
lobf_x = np.array((
    min(reciprocal_length) - MARGIN,
    max(reciprocal_length) + MARGIN
))

# Linear equation: y = mx + c
lobf_y = lobf_m * lobf_x + lobf_c

# Setup plotting
A4_SIZE = (11.69, 8.27)  # (x, y)
FIGURE_NAME = "Stationary_Waves.png"
plt.figure(figsize=A4_SIZE)

# Set appropriate labels
plt.xlabel(r"Length, 1/L $(m^{-1})$")
plt.ylabel("Frequency, f (Hz)")
plt.title(
    "Investigating how the frequency of the first harmonic changes with length"
)

# Plot `points` and `lobf`.
plt.scatter(reciprocal_length, frequency, marker="x", color="Black")
plt.plot(lobf_x, lobf_y)

plt.grid()  # Set grid (stylistic choice).
plt.margins(x=0)  # Removing margins as per Mr. Awuah.
plt.savefig(FIGURE_NAME)

# Expected output:
#
# Plotting points: ((3.3333333333333335, 250.0), (2.5, 200.0), (2.0, 166.7), (1.6666666666666667, 125.0), (1.4285714285714286, 111.1), (1.25, 83.33))
# LOBF gradient (m): 78.68128660180521
# LOBF y-intercept (c): -3.6826114953308036
# LOBF equation: y = 78.681x + -3.683
#
