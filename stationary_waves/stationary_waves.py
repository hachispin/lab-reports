"""
Required Practical 1: "Investigating Stationary Waves".

Code for investigating graphical relation of the
frequency of the first harmonic against length.

Lab report due December 17th, 2025. for Mr. Awuah.
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial

# The data gathered and recorded in my lab book.
POINTS = (
    # (length, frequency)
    (0.3, 250.0),
    (0.4, 200.0),
    (0.5, 166.7),
    (0.6, 125.0),
    (0.7, 111.1),
    (0.8, 83.33),
)

# Unpack `POINTS` to x, y arrays.
x_values, y_values = map(np.array, zip(*POINTS))
print(f"Plotting points: {POINTS}")

# Find the degree 1 (linear) LOBF's coefficients.
lobf_c, lobf_m = Polynomial.fit(x_values, y_values, deg=1).convert().coef
print(f"LOBF gradient (m): {lobf_m}")
print(f"LOBF y-intercept (c): {lobf_c}")
print(f"LOBF equation: y = {lobf_m:.3f}x + {lobf_c:.3f}")

# Extend the LOBF by `MARGIN`.
#
# I've done this so the points (marked with crosses)
# don't get cut off when margins are set to zero.
MARGIN = 0.05
lobf_x_values = np.array((min(x_values) - MARGIN, max(x_values) + MARGIN))

# Linear equation: y = mx + c
lobf = lobf_m * lobf_x_values + lobf_c

# Set appropriate labels
plt.title("Investigating how the frequency of the first harmonic changes with length")
plt.xlabel("Length (m)")
plt.ylabel("Frequency (Hz)")

# Plot `POINTS` and `lobf`.
plt.scatter(x_values, y_values, marker="x", color="Black")
plt.plot(lobf_x_values, lobf)

plt.grid()  # Set grid (stylistic choice).
plt.margins(0)  # Removing margins as per Mr. Awuah.

# Show the plot.
plt.show()

# Expected output:
#
# Plotting points: ((0.3, 250.0), (0.4, 200.0), (0.5, 166.7), (0.6, 125.0), (0.7, 111.1), (0.8, 83.33))
# LOBF gradient (m): -326.2142857142857
# LOBF y-intercept (c): 335.43952380952385
# LOBF equation: y = -326.214x + 335.440
#
