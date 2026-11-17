"""
===============================================================================
PHYS 435 Stellar Model Project -- Lane-Emden solver with a hand-written RK4
===============================================================================

WHAT THIS FILE DOES
-------------------
Numerically solves the Lane-Emden equation for a polytrope of index n,

        1/xi^2 d/dxi ( xi^2 dtheta/dxi ) = -theta^n,
        theta(0) = 1,   theta'(0) = 0,

and finds the surface of the star, xi_1 (the first zero of theta), plus the
surface quantity  -xi_1^2 theta'(xi_1)  that is needed later to convert the
dimensionless solution into physical mass.

    xi     = dimensionless radius        (r = alpha * xi)
    theta  = dimensionless "temperature" (density rho = rho_c * theta^n)

HOW THE SOLVE WORKS (overview -- each piece is explained where it's coded)
--------------------------------------------------------------------------
  Piece 1. Rewrite the 2nd-order equation as two 1st-order equations,
           because RK4 advances first-order systems.
  Piece 2. Start just off-centre (xi0 = 1e-3) using a power series, because
           the equation contains 2/xi and cannot be evaluated at xi = 0.
  Piece 3. Advance the solution one step at a time with classical RK4.
  Piece 4. Keep stepping outward until theta drops to zero (the surface).
  Piece 5. Pin down xi_1 precisely by interpolating between the last two
           steps, since theta crosses zero *between* steps, not on one.
  Tests.   Check each piece against known answers (analytic solutions for
           n = 0 and n = 1, and the published xi_1 for n = 3).



HOW TO RUN
----------
    python lane_emden_rk4.py

This runs the tests and prints a table of xi_1 and -xi_1^2 theta'(xi_1).
Requires only NumPy.
"""

import numpy as np


# =============================================================================
# PIECE 1: The right-hand side F(xi, Y)
# =============================================================================
#
# WHERE THE EQUATIONS COME FROM
# Expanding the Lane-Emden equation with the product rule (general knowledge):
#
#     d/dxi( xi^2 theta' ) = 2 xi theta' + xi^2 theta''
#
# and dividing by xi^2 gives the expanded form
#
#     theta'' + (2/xi) theta' + theta^n = 0.                          (*)
#
# RK4 only knows how to advance FIRST-order equations of the form
# dY/dxi = F(xi, Y). The standard trick is to promote the first derivative to
# its own variable:
#
#     y = theta,     z = dtheta/dxi
#
# Then:
#     dy/dxi = z                     (just the definition of z)
#     dz/dxi = theta'' = -2z/xi - y^n   (solve (*) for theta'')
#
# This is exactly the system coded in [Zingale-shoot]'s right-hand-side
# function, and the same reduction to two coupled first-order equations is
# used in [Anandaram].
#
# We store the two unknowns together as one array, Y = [y, z], and this
# function returns the two derivatives together as F = [dy/dxi, dz/dxi].
# Packing them into arrays lets the RK4 step (Piece 3) treat the whole system
# with ordinary vector arithmetic -- [Falk] notes that for a system, Y, F and
# each k_i simply become vectors.
# -----------------------------------------------------------------------------
def lane_emden_rhs(xi, Y, n):
    """Return F(xi, Y) = [dy/dxi, dz/dxi] for the state Y = [y, z].

    Parameters
    ----------
    xi : float   current dimensionless radius (must be > 0, see Piece 2)
    Y  : array   current state [y, z] = [theta, dtheta/dxi]
    n  : float   polytropic index
    """
    # Unpack the state vector into its two components.
    y, z = Y

    # SURFACE GUARD
    # Near the surface theta ~ 0. Inside a single RK4 step, the intermediate
    # evaluations (k2, k3, k4 in Piece 3) look at trial states like Y + k1/2,
    # which can have y slightly NEGATIVE even when the current y is positive.
    #
    # For non-integer n (e.g. n = 1.5), a negative number raised to a
    # fractional power is not a real number: plain Python returns a complex
    # number and NumPy returns nan (general knowledge). A single nan would
    # then poison every later value.
    #
    # Physically, theta < 0 has no meaning: density is rho = rho_c theta^n,
    # and the star simply ends where theta = 0 (zero density and pressure).
    # So we treat any negative trial y as zero before raising it to n.
    # This only ever affects the final step that crosses the surface, and
    # Piece 5 handles that step by interpolation anyway.
    y_pos = max(y, 0.0)

    # First equation: dy/dxi = z, by definition of z.
    dy_dxi = z

    # Second equation: dz/dxi = theta'' = -(2/xi) z - y^n,
    # from solving the expanded equation (*) for theta''.
    dz_dxi = -2.0 * z / xi - y_pos**n

    return np.array([dy_dxi, dz_dxi])


# =============================================================================
# PIECE 2: Starting values from the central power series
# =============================================================================
#
# WHY WE CAN'T START AT xi = 0
# At the centre z(0) = 0 and xi = 0, so the term 2z/xi in Piece 1 is 0/0.
# A computer would divide by zero. [Hunter] identifies xi = 0 as a singular
# point of the Lane-Emden equation and notes that a series solution is needed
# there.
#
# THE FIX
# Start a tiny distance out, at xi0, and get theta and theta' there from a
# power series that is valid near the centre. Derivation (your Task 4):
#
#   Try theta = a0 + a1 xi + a2 xi^2 + a3 xi^3 + a4 xi^4 + ...
#   - theta(0) = 1           =>  a0 = 1
#   - theta'(0) = 0          =>  a1 = 0
#   - Substitute into (*), using the binomial expansion
#     (1 + eps)^n ~ 1 + n*eps for theta^n, and set each power of xi to zero:
#         xi^0:  6 a2 + 1 = 0          =>  a2 = -1/6
#         xi^1:  12 a3 = 0             =>  a3 = 0
#         xi^2:  20 a4 + n a2 = 0      =>  a4 = n/120
#
#   Result:  theta(xi)  ~ 1 - xi^2/6 + n xi^4/120
#   and differentiating term by term:
#            theta'(xi) ~ -xi/3 + n xi^3/30
#
# This method (binomial expansion of theta^n, then grouping powers of xi) is
# the one set out in [Zingale-HW6], which also gives the same final series;
# [Seidov] gives the same expansion (writing n/120 as n/5!). [Hunter] shows
# the series contains only even powers of xi, consistent with a1 = a3 = 0.
# Using the series to initialise the solver at a small xi_min is also how
# [Pisces] does it.
#
# HOW SMALL SHOULD xi0 BE?
# The error of the truncated series is the first dropped term, which is
# order xi^6 (odd powers vanish). At xi0 = 1e-3, xi0^6 = 1e-18 -- below the
# ~1e-16 precision of a double-precision float (general knowledge). So the
# series is effectively exact there. [Pisces] uses a default of 1e-5; any
# value in that range works.
#
# A caution worth one sentence in the report: [Hunter] shows the series
# about the centre does NOT converge all the way to the surface for
# n > 1.9121. That's why we only use it to START the integration and let
# RK4 do the rest.
# -----------------------------------------------------------------------------
def series_start(xi0, n):
    """Return the starting state [y(xi0), z(xi0)] from the central series."""
    # theta(xi0) ~ 1 - xi0^2/6 + n xi0^4/120
    y0 = 1.0 - xi0**2 / 6.0 + n * xi0**4 / 120.0

    # theta'(xi0): derivative of the line above, term by term:
    #   d/dxi(-xi^2/6)     = -2xi/6   = -xi/3
    #   d/dxi(n xi^4/120)  = 4n xi^3/120 = n xi^3/30
    z0 = -xi0 / 3.0 + n * xi0**3 / 30.0

    return np.array([y0, z0])


# =============================================================================
# PIECE 3: One step of classical fourth-order Runge-Kutta (RK4)
# =============================================================================
#
# THE IDEA
# Euler's method steps forward using only the slope at the start of the step,
# Y_new = Y + h F(xi, Y), which is crude. RK4 instead samples the slope four
# times across the step and takes a weighted average, which makes it much
# more accurate for the same step size.
#
# THE FORMULAS (classical RK4, as given in [Falk]):
#
#     k1 = h F(xi,       Y         )   slope at the start of the step
#     k2 = h F(xi + h/2, Y + k1/2  )   slope at the midpoint, using k1 to get there
#     k3 = h F(xi + h/2, Y + k2/2  )   slope at the midpoint again, using k2
#     k4 = h F(xi + h,   Y + k3    )   slope at the end, using k3 to get there
#
#     Y_new = Y + (k1 + 2 k2 + 2 k3 + k4) / 6
#
# The weights 1, 2, 2, 1 (divided by 6) give the two midpoint estimates double
# weight. [Oxford] describes this as the classic four-stage, fourth-order
# explicit Runge-Kutta method.
#
# "FOURTH ORDER" means the global error scales like h^4: halving the step
# size should cut the error by about 2^4 = 16. That is the basis of the
# convergence test in Step 2 of the project.
#
# FOR A SYSTEM: because Y = [y, z] is an array, each k is also a two-component
# array -- one entry for y, one for z -- and NumPy does the arithmetic on both
# components at once. [Falk] states that for systems, Y, F and the k_i simply
# become vectors.
# -----------------------------------------------------------------------------
def rk4_step(f, xi, Y, h, n):
    """Advance the state Y from xi to xi + h with one classical RK4 step.

    Parameters
    ----------
    f  : function   right-hand side, called as f(xi, Y, n) -> array
    xi : float      current position
    Y  : array      current state [y, z]
    h  : float      step size
    n  : float      polytropic index (passed straight through to f)
    """
    # Slope at the start of the step.
    k1 = h * f(xi, Y, n)

    # Slope at the midpoint, reached by stepping halfway along k1.
    k2 = h * f(xi + h / 2.0, Y + k1 / 2.0, n)

    # Slope at the midpoint again, this time reached using the better
    # estimate k2.
    k3 = h * f(xi + h / 2.0, Y + k2 / 2.0, n)

    # Slope at the end of the step, reached by stepping the full way along k3.
    k4 = h * f(xi + h, Y + k3, n)

    # Weighted average of the four slope estimates (weights 1, 2, 2, 1).
    return Y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# =============================================================================
# PIECE 4: Integrate outward until the surface
# =============================================================================
#
# Start at xi0 with the series values (Piece 2), then repeatedly apply the
# RK4 step (Piece 3) with a fixed step size h, recording every point. This is
# the same "integrate outward from the centre until theta reaches zero"
# approach used throughout the polytrope literature; the first zero xi_1 is
# the stellar surface.
#
# STOPPING CONDITION: keep going while y > 0 (still inside the star). The
# first step that produces y <= 0 has just crossed the surface; we keep that
# point too, because Piece 5 needs the points on BOTH sides of the surface.
#
# SAFETY STOP (xi_max): for n >= 5 theta never reaches zero (the n = 5
# polytrope has infinite radius -- see [Seidov]/[Chandrasekhar]), so without
# a cap the loop would run forever. For n just below 5, xi_1 gets very large
# (around 32 for n = 4.5), so xi_max must be generous.
#
# STEP SIZE: h = 1e-3 is a choice, not a law. For n = 3 (xi_1 ~ 6.9) that is
# about 7000 steps, which runs in well under a second. The convergence test
# (halve h, check the error drops ~16x) is what justifies the choice in the
# report.
# -----------------------------------------------------------------------------
def integrate_to_surface(n, h=1e-3, xi0=1e-3, xi_max=200.0):
    """Step outward from xi0 until theta <= 0 (or xi_max is reached).

    Returns three arrays -- xi, y (= theta), z (= theta') -- for every point,
    including the final point just past the surface.
    """
    # Initial position and state, from the central series (Piece 2).
    xi = xi0
    Y = series_start(xi0, n)

    # Lists to record the solution at every step (lists grow cheaply;
    # we convert to arrays at the end).
    xi_list = [xi]
    y_list = [Y[0]]
    z_list = [Y[1]]

    # March outward while still inside the star and below the safety cap.
    while Y[0] > 0 and xi < xi_max:
        Y = rk4_step(lane_emden_rhs, xi, Y, h, n)   # advance one step
        xi = xi + h                                 # move the position on
        xi_list.append(xi)
        y_list.append(Y[0])
        z_list.append(Y[1])

    return np.array(xi_list), np.array(y_list), np.array(z_list)


# =============================================================================
# PIECE 5: Locate the surface xi_1 by linear interpolation
# =============================================================================
#
# THE PROBLEM: theta crosses zero somewhere BETWEEN the last two points, not
# exactly on one. Just taking the last xi would make xi_1 wrong by up to one
# step size h.
#
# THE FIX (linear interpolation, general knowledge): over one tiny step, theta
# is very nearly a straight line. The last two points bracket the surface:
#
#     (xi_a, y_a) with y_a > 0      -- last point inside the star
#     (xi_b, y_b) with y_b <= 0     -- first point outside
#
# The straight line through them is  y(xi) = y_a + (y_b - y_a)(xi - xi_a)/(xi_b - xi_a).
# Setting y = 0 and solving:
#
#     frac = (xi_1 - xi_a)/(xi_b - xi_a) = y_a / (y_a - y_b)
#     xi_1 = xi_a + frac * (xi_b - xi_a)
#
# "frac" is the fraction of the way across the last step where the line hits
# zero. Since y_a > 0 and y_b <= 0, frac is always between 0 and 1.
#
# We interpolate z = theta' to the same point using the same frac, giving
# theta'(xi_1).
#
# WHY WE NEED theta'(xi_1): [Zingale-shoot] notes that the combination
#     -xi_1^2 theta'(xi_1)
# is commonly needed to get physical values -- it sets the total mass of the
# polytrope, which the web app needs for its M slider.
#
# ACCURACY: linear interpolation has an error of order h^2 (general
# knowledge). With h = 1e-3 that's ~1e-6 or better, which is plenty against
# the 4-5 significant figures of published xi_1 tables.
# -----------------------------------------------------------------------------
def find_surface(xi, y, z):
    """Return (xi_1, theta'(xi_1)) by interpolating across the sign change."""
    # If the last point is still positive, we never reached the surface
    # (n >= 5, or xi_max too small).
    if y[-1] > 0:
        raise ValueError("Integration never reached the surface "
                         "(n >= 5, or xi_max too small).")

    # The two points that bracket the surface.
    xi_a, xi_b = xi[-2], xi[-1]   # positions
    y_a, y_b = y[-2], y[-1]       # theta values: y_a > 0 >= y_b
    z_a, z_b = z[-2], z[-1]       # theta' values

    # Fraction of the way from xi_a to xi_b where theta hits zero.
    frac = y_a / (y_a - y_b)

    # The surface position.
    xi_1 = xi_a + frac * (xi_b - xi_a)

    # theta' at the surface, interpolated with the same fraction.
    dtheta_1 = z_a + frac * (z_b - z_a)

    return xi_1, dtheta_1


def solve_lane_emden(n, h=1e-3, xi0=1e-3):
    """Full solve for one polytropic index n.

    Returns
    -------
    xi, y, z  : arrays   the solution profile (y = theta, z = theta')
    xi_1      : float    surface position (first zero of theta)
    dtheta_1  : float    theta'(xi_1), slope at the surface
    """
    xi, y, z = integrate_to_surface(n, h=h, xi0=xi0)
    xi_1, dtheta_1 = find_surface(xi, y, z)
    return xi, y, z, xi_1, dtheta_1


# =============================================================================
# TESTS
# =============================================================================
#
# Each piece is checked against an answer we know independently:
#   Piece 1: hand-calculated values of F.
#   Piece 2: the exact solutions theta = 1 - xi^2/6 (n = 0) and
#            theta = sin(xi)/xi (n = 1), both listed in [Seidov] citing
#            [Chandrasekhar].
#   Piece 3: one step on y' = y, whose exact answer is e^h. RK4's error for a
#            single step is of order h^5, so it should match to ~1e-7.
#   Pieces 4-5: xi_1 = sqrt(6) for n = 0 and pi for n = 1 [Seidov], and
#            xi_1 ~ 6.8968 for n = 3 [Anandaram].
# =============================================================================
def _check(name, got, expected, tol):
    """Print PASS/FAIL comparing got to expected within an absolute tolerance."""
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    ok = np.allclose(got, expected, atol=tol, rtol=0)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, expected {expected}")
    return ok


def _run_tests():
    print("Piece 1: lane_emden_rhs  (hand-calculated values)")
    _check("xi=1, Y=[1, 0],     n=1", lane_emden_rhs(1, [1, 0], 1), [0, -1], 1e-12)
    _check("xi=2, Y=[1, -1],    n=0", lane_emden_rhs(2, [1, -1], 0), [-1, 0], 1e-12)
    _check("xi=1, Y=[0.5,-0.2], n=2", lane_emden_rhs(1, [0.5, -0.2], 2), [-0.2, 0.15], 1e-12)

    print("\nPiece 2: series_start  (vs exact n=0 and n=1 solutions at xi=0.1)")
    x = 0.1
    # n = 0: theta = 1 - xi^2/6 exactly, so theta' = -xi/3. Series is exact.
    _check("n=0", series_start(x, 0), [1 - x**2 / 6, -x / 3], 1e-12)
    # n = 1: theta = sin(xi)/xi, theta' = (xi cos xi - sin xi)/xi^2.
    # Series is truncated, so agreement is only to the size of the next term.
    _check("n=1", series_start(x, 1),
           [np.sin(x) / x, (x * np.cos(x) - np.sin(x)) / x**2], 1e-7)

    print("\nPiece 3: rk4_step  (one step on y' = y, exact answer e^h)")
    f_exp = lambda xi, Y, n: np.array([Y[0], 0.0])
    h = 0.1
    _check("h=0.1", rk4_step(f_exp, 0.0, np.array([1.0, 0.0]), h, 0)[0], np.exp(h), 1e-6)

    print("\nPieces 4-5: surface xi_1 vs known values")
    _check("n=0  (sqrt 6)",  solve_lane_emden(0)[3], np.sqrt(6), 1e-5)
    _check("n=1  (pi)",      solve_lane_emden(1)[3], np.pi, 1e-5)
    _check("n=3  (~6.8968)", solve_lane_emden(3)[3], 6.8968, 1e-4)

    print("\nResults table:")
    print("   n      xi_1        -xi_1^2 theta'(xi_1)")
    for n in [0, 1, 1.5, 2, 3, 4, 4.5]:
        *_, xi_1, dtheta_1 = solve_lane_emden(n)
        print(f"  {n:<5} {xi_1:10.5f}   {-xi_1**2 * dtheta_1:10.5f}")


if __name__ == "__main__":
    _run_tests()