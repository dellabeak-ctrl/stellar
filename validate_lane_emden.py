"""
===============================================================================
PHYS 435 Stellar Model Project -- Step 2: validating the RK4 Lane-Emden solver
===============================================================================

WHAT THIS FILE DOES
-------------------
Checks that the hand-written RK4 solver in lane_emden_rk4.py solves the
Lane-Emden equation correctly, three independent ways:

  Test A. AGAINST THE EXACT SOLUTIONS (n = 0, 1, 5)
          These are the only indices with closed-form solutions, so for them
          we can measure the true error of the solver directly.

  Test B. AGAINST SCIPY (n = 1.5, 2, 3, 4)
          For indices with no exact solution, compare against SciPy's
          solve_ivp run at very tight tolerance with a different, higher-order
          integrator. This is the "pre-existing polytrope code" check the prof
          suggested.

  Test C. CONVERGENCE (the order of the method)
          Halve the step size repeatedly. For a correct fourth-order method,
          the error should shrink by about 2^4 = 16 each time. Seeing that
          factor is strong evidence the RK4 is implemented correctly, not
          just "close enough".

It also makes the main results figure: theta(xi) for several n.

OUTPUT
------
Prints result tables to the terminal and saves four figures (PNG) in the
same folder as this script:

    fig1_profiles.png       theta(xi) vs xi/xi_1 for n = 0, 1, 1.5, 2, 3, 4
    fig2_exact_error.png    |RK4 - exact| for n = 0, 1, 5        (Test A)
    fig3_scipy_diff.png     |RK4 - SciPy| for n = 1.5, 3, 4      (Test B)
    fig4_convergence.png    error vs step size h, log-log        (Test C)

HOW TO RUN
----------
Put this file in the SAME folder as lane_emden_rk4.py (it imports from it),
then:
    python validate_lane_emden.py
Needs NumPy, SciPy and Matplotlib:
    python -m pip install numpy scipy matplotlib

REFERENCES (short tags used in the comments)
--------------------------------------------
  [SciPy]      SciPy documentation, scipy.integrate.solve_ivp.
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
  [Zingale-shoot]  M. Zingale, "Solving Lane-Emden via Shooting" -- solves
               Lane-Emden with SciPy's solve_ivp (RK45, tolerances 1e-8).
               https://zingale.github.io/stars/notebooks/polytrope/lane-emden-shooting.html
  [Anandaram]  M. N. Anandaram, "Evaluation of Scipy.ode Integrators in
               Solving the Lane-Emden Equation...", Mapana J. Sci. -- finds
               SciPy's dopri5/dop853 integrators accurate to ~14 decimal
               digits on this problem; gives xi_1 ~ 6.8968 for n = 3.
               https://journals.christuniversity.in/index.php/mapana/article/view/1478
  [Seidov]     Z. Seidov, arXiv:astro-ph/0402130 -- exact solutions for
               n = 0 and n = 1 and their xi_1, citing Chandrasekhar.
  [Sharaf]     M. A. Sharaf & A. N. Al-Amoudi, power-series solution of the
               Lane-Emden equation -- lists the three exact solutions,
               including n = 5.
               https://marz.kau.edu.sa/Files/1001002/Researches/66071_37597.pdf
  [Chandrasekhar] S. Chandrasekhar (1939), An Introduction to the Study of
               Stellar Structure, Ch. IV -- textbook source for all of the
               exact solutions.
  [Falk]       R. Falk, MATH 573 lecture notes (Rutgers) -- classical RK4.
  [Oxford]     Oxford Numerical Analysis Lectures 14-15 -- RK4 is a
               four-stage, fourth-order method.
  "(general knowledge)" marks standard numerical-methods facts with no single
  source; any numerical analysis textbook covers them.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # draw to files, no pop-up windows needed
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Pieces of the solver from Step 1.
from lane_emden_rk4 import (
    lane_emden_rhs,
    series_start,
    rk4_step,
    solve_lane_emden,
)

# Figures are saved next to this script (e.g. on your Desktop).
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Starting point and default step size -- the same values used in Step 1.
XI0 = 1e-3
H_DEFAULT = 1e-3


# =============================================================================
# THE EXACT SOLUTIONS (used by Tests A and C)
# =============================================================================
#
# Only three indices have closed-form solutions [Seidov], [Sharaf],
# [Chandrasekhar]:
#
#   n = 0:  theta = 1 - xi^2/6           surface xi_1 = sqrt(6)
#   n = 1:  theta = sin(xi)/xi           surface xi_1 = pi
#   n = 5:  theta = (1 + xi^2/3)^(-1/2)  never reaches zero (infinite radius)
#
# Their derivatives (general knowledge, by differentiating):
#   n = 0:  theta' = -xi/3
#   n = 1:  theta' = (xi cos xi - sin xi)/xi^2
#   n = 5:  theta' = -(xi/3) (1 + xi^2/3)^(-3/2)
# -----------------------------------------------------------------------------
def theta_exact(xi, n):
    """Exact Lane-Emden solution theta(xi) for n = 0, 1 or 5."""
    xi = np.asarray(xi, dtype=float)
    if n == 0:
        return 1.0 - xi**2 / 6.0
    if n == 1:
        return np.sin(xi) / xi
    if n == 5:
        return (1.0 + xi**2 / 3.0) ** -0.5
    raise ValueError("Exact solutions exist only for n = 0, 1, 5.")


# =============================================================================
# HELPER: fixed number of RK4 steps (used by Tests A and C)
# =============================================================================
#
# solve_lane_emden() from Step 1 always runs to the surface. For Tests A and C
# we instead want to integrate to a CHOSEN point, for two reasons:
#   - n = 5 has no surface, so "run to the surface" never stops.
#   - The convergence test (C) must compare every step size at exactly the
#     same xi, so we need the grid to land precisely on that point.
#
# The grid is xi_k = XI0 + k*h. We compute xi_k by multiplication rather than
# by repeatedly adding h, because adding h thousands of times accumulates
# floating-point rounding error (general knowledge), which would shift the
# final point slightly and contaminate the convergence test.
# -----------------------------------------------------------------------------
def rk4_fixed_steps(n, h, n_steps, xi0=XI0):
    """Take exactly n_steps RK4 steps of size h from xi0.

    Returns arrays xi (length n_steps+1) and y = theta at each point.
    """
    Y = series_start(xi0, n)                 # central series start (Step 1)
    xi_arr = xi0 + h * np.arange(n_steps + 1)
    y_arr = np.empty(n_steps + 1)
    y_arr[0] = Y[0]
    for k in range(n_steps):
        Y = rk4_step(lane_emden_rhs, xi_arr[k], Y, h, n)
        y_arr[k + 1] = Y[0]
    return xi_arr, y_arr


# =============================================================================
# HELPER: the SciPy reference solution (used by Test B)
# =============================================================================
#
# solve_ivp [SciPy] is SciPy's general ODE integrator. It is the same tool
# [Zingale-shoot] uses for Lane-Emden. Choices made here:
#
#   method="DOP853"  An explicit Runge-Kutta method of order 8 [SciPy] -- a
#                    DIFFERENT and higher-order method than our RK4, so
#                    agreement isn't just one RK4 agreeing with itself.
#                    [Anandaram] finds SciPy's dop853 accurate to about 14
#                    decimal digits on the Lane-Emden problem.
#   rtol, atol=1e-12 Very tight error tolerances, so the reference is far
#                    more accurate than the RK4 run we're checking.
#                    (Zingale uses 1e-8; we go tighter because this is the
#                    reference, not the thing being tested.)
#   adaptive steps   solve_ivp chooses its own step sizes to meet the
#                    tolerances [SciPy] -- another way it differs from our
#                    fixed-step RK4.
#   events           An "event" function makes solve_ivp locate where a
#                    quantity crosses zero and, if marked terminal, stop
#                    there [SciPy]. Our event is y = theta, so it finds the
#                    surface xi_1 directly, with its own root-finding --
#                    independent of our linear interpolation.
#   dense_output     Gives a continuous interpolant of the solution [SciPy],
#                    so we can evaluate SciPy's answer at exactly our RK4
#                    grid points for a point-by-point comparison.
#
# The same series start (Piece 2) and the same right-hand side (Piece 1) are
# used, because both solvers need to start off-centre and solve the same
# equation; what's being compared is the INTEGRATOR.
# -----------------------------------------------------------------------------
def scipy_reference(n, xi_max=200.0):
    """Solve Lane-Emden with SciPy's DOP853 at tight tolerance.

    Returns (solution object, xi_1 from SciPy's event finder).
    """
    def surface(xi, Y, n):
        return Y[0]                  # zero exactly where theta = 0
    surface.terminal = True          # stop integrating at the surface
    surface.direction = -1           # only count theta going + to -

    sol = solve_ivp(
        lane_emden_rhs,
        (XI0, xi_max),
        series_start(XI0, n),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        events=surface,
        dense_output=True,
        args=(n,),
    )
    xi_1 = sol.t_events[0][0]
    return sol, xi_1


# =============================================================================
# FIGURE STYLE
# =============================================================================
# Thin lines, light gridlines, muted axes, so the data stands out. Colours:
#   - Figure 1 encodes ORDERED values of n, so it uses one hue (blue) from
#     light to dark: darker = larger n.
#   - Figures 2-4 compare a few separate cases, so they use three clearly
#     different hues (blue, orange, aqua), chosen to stay distinguishable for
#     colour-blind readers. Every series is also labelled.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]                       # blue, orange, aqua
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # light -> dark

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "font.size": 10,
    "legend.frameon": False,
})


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# =============================================================================
# FIGURE 1: theta profiles for several n
# =============================================================================
#
# Plotted against xi/xi_1 (fraction of the stellar radius), so every star
# runs from 0 (centre) to 1 (surface). Plotting against raw xi would stretch
# the n = 4 curve out to xi ~ 15 and squash the rest. Higher n should show a
# steeper central drop: more of the star's "stuff" concentrated at the
# centre -- this is the central-concentration trend from Step 3 of the plan.
# -----------------------------------------------------------------------------
def figure_profiles():
    print("\nFigure 1: theta profiles")
    ns = [0, 1, 1.5, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for n, colour in zip(ns, SEQ):
        xi, y, z, xi_1, _ = solve_lane_emden(n, h=H_DEFAULT)
        inside = xi <= xi_1                       # drop the point past the surface
        x = xi[inside] / xi_1
        ax.plot(x, y[inside], color=colour, label=f"n = {n}")
        # Direct label near each curve, at x = 0.5.
        i = np.searchsorted(x, 0.5)
        ax.annotate(f"n={n}", (x[i], y[inside][i]), xytext=(4, 2),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel(r"$\xi/\xi_1$  (fraction of stellar radius)")
    ax.set_ylabel(r"$\theta$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(r"Lane–Emden solutions $\theta(\xi)$ (own RK4, $h=10^{-3}$)",
                 loc="left", color=INK)
    ax.legend(loc="upper right")
    save(fig, "fig1_profiles.png")


# =============================================================================
# TEST A: error against the exact solutions (n = 0, 1, 5)
# =============================================================================
#
# For each exact case, integrate with the default step size and compute the
# absolute error |theta_RK4 - theta_exact| at every grid point.
#   n = 0, 1: integrate to just inside the surface (sqrt 6, pi).
#   n = 5:    no surface, so integrate to xi = 20.
# We also check xi_1 for n = 0, 1 against sqrt(6) and pi.
#
# WHAT TO EXPECT: errors around 1e-15 to 1e-14 for all three cases. That is
# floating-point round-off level (double precision carries ~16 significant
# digits, general knowledge): at h = 1e-3, RK4's own truncation error is
# even smaller than the round-off, so what you see is round-off, and the
# jagged shape of the curves reflects that. The xi_1 comparison is less
# precise (~1e-7) -- see the note in Test B: that comes from the linear
# interpolation, not from RK4.
# -----------------------------------------------------------------------------
def test_exact():
    print("\nTest A: RK4 vs exact solutions")
    cases = [(0, np.sqrt(6) - 0.01), (1, np.pi - 0.01), (5, 20.0)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    print("   n    max |theta_RK4 - theta_exact|")
    for (n, xi_end), colour in zip(cases, CAT):
        n_steps = int(round((xi_end - XI0) / H_DEFAULT))
        xi, y = rk4_fixed_steps(n, H_DEFAULT, n_steps)
        err = np.abs(y - theta_exact(xi, n))
        print(f"   {n}    {err.max():.3e}")
        # Floor at 1e-17 so exact zeros can still be drawn on a log axis.
        ax.semilogy(xi, np.maximum(err, 1e-17), color=colour, label=f"n = {n}")

    print("\n   n    xi_1 (RK4)          exact             |difference|")
    for n, exact in [(0, np.sqrt(6)), (1, np.pi)]:
        xi_1 = solve_lane_emden(n, h=H_DEFAULT)[3]
        print(f"   {n}    {xi_1:.12f}   {exact:.12f}   {abs(xi_1 - exact):.2e}")

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$|\theta_{\rm RK4} - \theta_{\rm exact}|$")
    ax.set_title(r"Test A: error vs exact solutions ($h=10^{-3}$)",
                 loc="left", color=INK)
    ax.legend()
    save(fig, "fig2_exact_error.png")


# =============================================================================
# TEST B: comparison against SciPy (indices with no exact solution)
# =============================================================================
#
# For each n, run our RK4 and SciPy's DOP853 reference, then compare:
#   - the surface xi_1 (our linear interpolation vs SciPy's event finder),
#   - the whole profile, by evaluating SciPy's dense output at our grid.
#
# WHAT TO EXPECT: profile differences around 1e-12. Agreement between two
# different integrators (fixed-step 4th order vs adaptive 8th order) means
# both are solving the equation correctly. At this level the difference is
# set mostly by SciPy's own tolerance (1e-12), not by our RK4 -- Test A
# showed our RK4 error is ~1e-14.
#
# A NOTE ON xi_1: our xi_1 comes from LINEAR interpolation across the last
# step, which has an error of order h^2 (general knowledge) -- about 1e-7 at
# h = 1e-3. So xi_1 agrees slightly less closely than the profile itself,
# which only carries RK4's much smaller h^4 error. Worth a sentence in the
# report: the interpolation, not the RK4, limits the xi_1 accuracy.
# -----------------------------------------------------------------------------
def test_scipy():
    print("\nTest B: RK4 vs SciPy solve_ivp (DOP853, rtol = atol = 1e-12)")
    ns = [1.5, 2, 3, 4]
    print("   n     xi_1 (RK4)       xi_1 (SciPy)     |dxi_1|     max |dtheta| over profile")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted = 0
    for n in ns:
        xi, y, z, xi_1_rk4, _ = solve_lane_emden(n, h=H_DEFAULT)
        sol, xi_1_sp = scipy_reference(n)
        inside = xi <= min(xi_1_rk4, xi_1_sp)
        diff = np.abs(y[inside] - sol.sol(xi[inside])[0])
        print(f"   {n:<4}  {xi_1_rk4:.10f}   {xi_1_sp:.10f}   "
              f"{abs(xi_1_rk4 - xi_1_sp):.2e}    {diff.max():.2e}")
        if n in (1.5, 3, 4):
            ax.semilogy(xi[inside] / xi_1_sp, np.maximum(diff, 1e-17),
                        color=CAT[plotted], label=f"n = {n}")
            plotted += 1

    ax.set_xlabel(r"$\xi/\xi_1$")
    ax.set_ylabel(r"$|\theta_{\rm RK4} - \theta_{\rm SciPy}|$")
    ax.set_title(r"Test B: own RK4 vs SciPy DOP853 ($h=10^{-3}$)",
                 loc="left", color=INK)
    ax.legend()
    save(fig, "fig3_scipy_diff.png")


# =============================================================================
# TEST C: convergence -- is it really fourth order?
# =============================================================================
#
# THEORY: RK4 is a fourth-order method [Oxford], meaning the global error at
# a fixed point scales like h^4 (general knowledge). So if we halve h, the
# error should drop by 2^4 = 16. Equivalently, on a log-log plot of error vs
# h, the points should lie on a line of slope 4.
#
# The "observed order" between two step sizes is (general knowledge)
#
#     p = log( err(h) / err(h/2) ) / log(2),
#
# which should come out close to 4.
#
# SETUP: integrate to a fixed endpoint and compare with the exact solution.
#   n = 1: endpoint xi = XI0 + 3.0 (inside the surface at pi)
#   n = 5: endpoint xi = XI0 + 10.0
# Every step size divides the distance evenly, so all runs end at exactly
# the same xi. We use exact-solution cases so the error is the TRUE error.
#
# WHY THESE STEP SIZES: start coarse (h = 0.2) and halve. If h gets too
# small, the error eventually hits floating-point round-off (~1e-14 here,
# general knowledge) and stops shrinking -- the line flattens. That is
# expected and is NOT a sign the method is wrong; just base the order on the
# points before the flattening.
# -----------------------------------------------------------------------------
def test_convergence():
    print("\nTest C: convergence (error at a fixed point vs step size h)")
    hs = 0.2 / 2.0 ** np.arange(7)            # 0.2, 0.1, ..., 0.003125
    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    for (n, length), colour in zip([(1, 3.0), (5, 10.0)], CAT):
        errs = []
        for h in hs:
            n_steps = int(round(length / h))
            xi, y = rk4_fixed_steps(n, h, n_steps)
            errs.append(abs(y[-1] - theta_exact(xi[-1], n)))
        errs = np.array(errs)

        print(f"\n   n = {n}, error at xi = {XI0 + length:.3f}")
        print("        h          error        ratio to previous   observed order p")
        for i, (h, e) in enumerate(zip(hs, errs)):
            if i == 0:
                print(f"   {h:9.6f}   {e:.3e}")
            else:
                ratio = errs[i - 1] / e
                print(f"   {h:9.6f}   {e:.3e}        {ratio:7.2f}            {np.log2(ratio):5.2f}")

        ax.loglog(hs, errs, "o-", color=colour, markersize=5, label=f"n = {n}")

    # Reference line of slope 4, anchored to the n = 1 data at the largest h.
    n_steps = int(round(3.0 / hs[0]))
    xi, y = rk4_fixed_steps(1, hs[0], n_steps)
    e0 = abs(y[-1] - theta_exact(xi[-1], 1))
    ax.loglog(hs, e0 * (hs / hs[0]) ** 4, "--", color=MUTED, linewidth=1.2,
              label=r"slope 4 ($\propto h^4$)")

    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel(r"$|\theta_{\rm RK4} - \theta_{\rm exact}|$ at fixed $\xi$")
    ax.set_title("Test C: convergence of RK4", loc="left", color=INK)
    ax.legend()
    save(fig, "fig4_convergence.png")


if __name__ == "__main__":
    figure_profiles()
    test_exact()
    test_scipy()
    test_convergence()
    print("\nDone.")
