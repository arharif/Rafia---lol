"""
ECE 491 - Python Lab 1
Introduction to Python and One-Dimensional (1-D) Convolution

This file contains all Python code for Part A and Part B.
It saves the required figures in the current directory.

The code uses NumPy and Matplotlib only and is separated by lab question.
"""

import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend: avoids GUI/runtime errors.

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART A.1
# =============================================================================

x = np.array([0., 0., 1., 1., 1., 1., 1., 1., 1., 1.])
k = np.arange(-2., 8., 1.)

# The default index for stem(x) is n = 0,...,9.
# Therefore the first plot is u[n-2].
# With k = -2,...,7, the same samples are u[n].
# Thus the delay is d = 2.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].stem(np.arange(len(x)), x, basefmt=" ")
axes[0].set_title("Figure 1: u[n-2]")
axes[0].set_xlabel("n")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.1, 1.2)

axes[1].stem(k, x, basefmt=" ")
axes[1].set_title("Figure 2: u[n]")
axes[1].set_xlabel("n")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.1, 1.2)

fig.tight_layout()
fig.savefig("Part_A1_steps.png", dpi=160)
plt.close(fig)

print("Part A.1:")
print("  Figure 1 = u[n-2]")
print("  Figure 2 = u[n]")
print("  d = 2")


# =============================================================================
# PART A.2
# =============================================================================

k = np.arange(-10, 21, 1)

# u[n]: zero for n < 0, one for n >= 0.
N = 10
M = 20
u_n = np.hstack([np.zeros(shape=(N)), np.ones(shape=(M+1))])

# u[n-5]: zero for n < 5, one for n >= 5.
# Over -10 <= n <= 20 there are 15 zeros (-10,...,4)
# and 16 ones (5,...,20).
N = 15
M = 15
u_n_minus_5 = np.hstack([np.zeros(shape=(N)), np.ones(shape=(M+1))])

fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)

axes[0].stem(k, u_n, basefmt=" ")
axes[0].set_title("u[n], -10 <= n <= 20")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.1, 1.2)

axes[1].stem(k, u_n_minus_5, basefmt=" ")
axes[1].set_title("u[n-5], -10 <= n <= 20")
axes[1].set_xlabel("n")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.1, 1.2)

fig.tight_layout()
fig.savefig("Part_A2_steps.png", dpi=160)
plt.close(fig)


# =============================================================================
# PART A.3
# =============================================================================

k = np.arange(-40, 81, 1)

x_n = 7*np.cos(0.1*k) + np.cos(0.95*k)

# Time shift: x[n-20] is obtained by replacing n by n-20.
x_n_minus_20 = (
    7*np.cos(0.1*(k-20))
    + np.cos(0.95*(k-20))
)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(k, x_n)
axes[0].set_title("x[n] = 7 cos(0.1 n) + cos(0.95 n)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

axes[1].plot(k, x_n_minus_20)
axes[1].set_title("x[n-20]")
axes[1].set_xlabel("n")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("Part_A3_signals.png", dpi=160)
plt.close(fig)


# =============================================================================
# PART B - helper for full-convolution index axes
# =============================================================================

def full_convolution_axis(k, h):
    """
    If the input x is indexed by k[0],...,k[-1] and h starts at n=0,
    np.convolve(x,h) in 'full' mode is indexed from k[0] through
    k[-1] + len(h) - 1.
    """
    return np.arange(int(k[0]), int(k[-1]) + len(h))


# =============================================================================
# PART B.2 - MOVING AVERAGE SYSTEM
# =============================================================================

h_avg = (1./5.) * np.array([1., 1., 1., 1., 1.])

# (a) x = 1 for k = 0,...,20
k_a = np.arange(0, 21, 1)
x_a = np.ones_like(k_a, dtype=float)
y_a = np.convolve(x_a, h_avg)
ky_a = full_convolution_axis(k_a, h_avg)

# (b) x = cos(0.1k), k = -40,...,80
k_b = np.arange(-40, 81, 1)
x_b = np.cos(0.1*k_b)
y_b = np.convolve(x_b, h_avg)
ky_b = full_convolution_axis(k_b, h_avg)

# (c) x = cos(0.95k), k = -40,...,80
x_c = np.cos(0.95*k_b)
y_c = np.convolve(x_c, h_avg)
ky_c = full_convolution_axis(k_b, h_avg)

# (d) signal from Part A.3
x_d = 7*np.cos(0.1*k_b) + np.cos(0.95*k_b)
y_d = np.convolve(x_d, h_avg)
ky_d = full_convolution_axis(k_b, h_avg)

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

for ax, kk, yy, title in [
    (axes[0,0], ky_a, y_a, "B.2(a): x[n] = 1"),
    (axes[0,1], ky_b, y_b, "B.2(b): x[n] = cos(0.1n)"),
    (axes[1,0], ky_c, y_c, "B.2(c): x[n] = cos(0.95n)"),
    (axes[1,1], ky_d, y_d, "B.2(d): mixed signal"),
]:
    ax.plot(kk, yy)
    ax.set_xlabel("n")
    ax.set_ylabel("y[n]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("Part_B2_moving_average.png", dpi=160)
plt.close(fig)

print("\\nPart B.2(a) moving-average output for finite constant input:")
print(y_a)


# =============================================================================
# PART B.3 - DIFFERENCE SYSTEM
# =============================================================================

h_diff = np.array([1., -1.])

# Reuse the four inputs above.
yd_a = np.convolve(x_a, h_diff)
kyd_a = full_convolution_axis(k_a, h_diff)

yd_b = np.convolve(x_b, h_diff)
kyd_b = full_convolution_axis(k_b, h_diff)

yd_c = np.convolve(x_c, h_diff)
kyd_c = full_convolution_axis(k_b, h_diff)

yd_d = np.convolve(x_d, h_diff)
kyd_d = full_convolution_axis(k_b, h_diff)

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

for ax, kk, yy, title in [
    (axes[0,0], kyd_a, yd_a, "B.3(a): x[n] = 1"),
    (axes[0,1], kyd_b, yd_b, "B.3(b): x[n] = cos(0.1n)"),
    (axes[1,0], kyd_c, yd_c, "B.3(c): x[n] = cos(0.95n)"),
    (axes[1,1], kyd_d, yd_d, "B.3(d): mixed signal"),
]:
    ax.plot(kk, yy)
    ax.set_xlabel("n")
    ax.set_ylabel("y[n]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("Part_B3_difference.png", dpi=160)
plt.close(fig)

print("\\nPart B.3(a) difference-system output for finite constant input:")
print(yd_a)


# =============================================================================
# PART B.4 - INTERPRETATION
# =============================================================================

def frequency_response_magnitude(h, omega):
    n = np.arange(len(h))
    return abs(np.sum(h*np.exp(-1j*omega*n)))

print("\\nPart B.4 - magnitude response at the two cosine frequencies:")
print("Moving average |H(0.1)|  =", frequency_response_magnitude(h_avg, 0.1))
print("Moving average |H(0.95)| =", frequency_response_magnitude(h_avg, 0.95))
print("Difference     |H(0.1)|  =", frequency_response_magnitude(h_diff, 0.1))
print("Difference     |H(0.95)| =", frequency_response_magnitude(h_diff, 0.95))

print("\\nInterpretation:")
print("  Moving average: smoothing / low-pass behavior; preserves DC and low frequency,")
print("  while attenuating higher-frequency oscillation.")
print("  Difference system: high-pass / first-difference behavior; removes DC and")
print("  emphasizes faster sample-to-sample changes.")
