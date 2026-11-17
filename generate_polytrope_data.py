"""
===============================================================================
PHYS 435 Stellar Model Project -- Step 3: generating the data for the web app
===============================================================================

WHAT THIS FILE DOES
-------------------
The web app needs Lane-Emden solutions for whatever polytropic index n the
user picks with the slider. Solving an ODE in the browser on every slider
move is possible but unnecessary, because of one key fact:

    The dimensionless solution theta(xi) depends ONLY on n.
    Mass M, radius R and mean molecular weight mu just RESCALE it.

So we solve once per n here, in Python with SciPy (the "use pre-existing code
under the hood" route the prof suggested), save the dimensionless results to a
JSON file, and let the web page do the cheap rescaling for M, R, mu.

This file:
  1. Solves Lane-Emden with SciPy for n = 0.00, 0.05, 0.10, ..., 4.50.
  2. Resamples each solution onto a common grid of x = xi/xi_1 = r/R.
  3. Saves everything to polytropes.json (the web app's input).
  4. Implements the dimensionless -> physical rescaling in Python. The web app
     will do the same maths in JavaScript, and this Python version is the
     reference to test the JavaScript against.
  5. Runs physics checks: total mass, hydrostatic equilibrium, the stored-grid
     resolution, and agreement with your own RK4 from Step 1.

HOW TO RUN
----------
Same folder as lane_emden_rk4.py, then:
    python generate_polytrope_data.py
Writes polytropes.json and polytropes.js (the same data, in the form the
web page loads) to that folder and prints the checks.

REFERENCES (short tags used below)
----------------------------------
  [SciPy]      SciPy docs, scipy.integrate.solve_ivp.
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
  [SciPy-const] SciPy docs, scipy.constants (CODATA physical constants).
               https://docs.scipy.org/doc/scipy/reference/constants.html
  [Anandaram]  M. N. Anandaram, Mapana J. Sci. -- SciPy's dop853 is accurate
               to ~14 digits on Lane-Emden.
  [WVU]        "Stellar Models" lecture notes, West Virginia University
               (L. Anderson's course page). Gives r = alpha*xi, the mass
               M = 4 pi alpha^3 rho_c (-xi_1^2 theta'(xi_1)), and the
               notation rho_c = D_n * rho_mean. The expression for D_n used
               here is derived from the mass formula in Part 3(a).
               https://lorenanderson.faculty.wvu.edu/files/d/08974f96-09aa-41d9-97f5-de4f2615568e/stellar_models.pdf
  [UMD]        M. C. Miller, ASTR 406 Lecture 23 (U. Maryland). Gives the
               radius scale  r_n^2 = (n+1) P_c / (4 pi G rho_c^2), and
               rho = rho_c theta^n, r = r_n xi.
               https://pages.astro.umd.edu/~mcmiller/teaching/astr406/lecture23.pdf
  [Caltech]    Ay 101 polytropes notes (Caltech). Gives the ideal-gas
               pressure P_gas = rho k T / (mu m_H), and the n = 3 Eddington
               relation used later for beta.
               https://sites.astro.caltech.edu/~jlc/ay101_fall2015/ay101_polytropes_fall2015.pdf
  [IAU-B3]     IAU 2015 Resolution B3 on nominal solar values (Prsa et al.
               2016, arXiv:1510.07674): nominal solar radius 6.957e8 m and
               nominal solar mass parameter GM = 1.3271244e20 m^3 s^-2.
               https://ar5iv.labs.arxiv.org/html/1510.07674
  [Chandrasekhar] S. Chandrasekhar (1939), Ch. IV -- textbook source for the
               polytrope relations and tables of constants.
  "(general knowledge)" marks standard facts with no single source.
"""

import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants as const

from lane_emden_rk4 import lane_emden_rhs, series_start, solve_lane_emden

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
XI0 = 1e-3     # off-centre starting point, same as Steps 1-2


# =============================================================================
# CHOICES FOR THE DATA GRID
# =============================================================================
#
# n GRID: 0.00 to 4.50 in steps of 0.05 (91 solutions).
#   - Step 0.05 is fine enough for smooth slider movement. The intended
#     design is that the slider SNAPS to these values, so the page always
#     shows an exact solution and never has to interpolate between two n.
#   - Upper limit 4.5, not 5: theta never reaches zero for n = 5 (infinite
#     radius [Chandrasekhar]), and xi_1 grows very quickly as n -> 5
#     (xi_1 ~ 32 already at n = 4.5, from Step 1). Real stars are usually
#     modelled with n between about 1.5 and 3, so 4.5 leaves plenty of room
#     (general knowledge -- worth a sentence in the report justifying it).
#
# x GRID: x = xi/xi_1 = r/R, from 0 (centre) to 1 (surface).
#   - Using x rather than xi means every n shares the same grid, and the
#     page can plot every star from centre to surface on the same axis.
#   - 401 evenly spaced points. High-n polytropes are very centrally
#     concentrated (theta drops steeply near x = 0), which is the hardest
#     case for a fixed grid. The resolution check at the bottom of this file
#     measures how accurately the page's linear interpolation between these
#     points reproduces the true solution, so this number is justified by a
#     measurement rather than a guess.
# -----------------------------------------------------------------------------
N_VALUES = np.round(np.arange(0.0, 4.5 + 1e-9, 0.05), 2)
X_GRID = np.linspace(0.0, 1.0, 401)


# =============================================================================
# PART 1: Solve Lane-Emden with SciPy for one n
# =============================================================================
#
# Same reference setup as Step 2 (see validate_lane_emden.py for the full
# explanation of each option):
#   - DOP853, an order-8 Runge-Kutta method, with rtol = atol = 1e-12 [SciPy];
#     [Anandaram] finds it accurate to ~14 digits on this equation.
#   - Start at XI0 with the central series (Step 1, Piece 2).
#   - A terminal "event" stops at theta = 0 and returns xi_1 [SciPy].
#   - dense_output gives a smooth interpolant, so we can evaluate the solution
#     at exactly the xi values we want [SciPy].
# -----------------------------------------------------------------------------
def solve_scipy(n):
    def surface(xi, Y, n):
        return Y[0]
    surface.terminal = True
    surface.direction = -1

    sol = solve_ivp(lane_emden_rhs, (XI0, 200.0), series_start(XI0, n),
                    method="DOP853", rtol=1e-12, atol=1e-12,
                    events=surface, dense_output=True, args=(n,))
    xi_1 = sol.t_events[0][0]
    dtheta_1 = sol.y_events[0][0][1]       # theta'(xi_1), returned at the event
    return sol, xi_1, dtheta_1


# =============================================================================
# PART 2: The dimensionless quantities the web app needs
# =============================================================================
#
# For each n we store two numbers and two profiles.
#
# NUMBERS
#   xi_1         the surface (first zero of theta).
#   dtheta_1     theta'(xi_1), the slope at the surface.
# From these two, every scaling constant follows (Part 4).
#
# PROFILES on the x grid
#   theta(x)     the Lane-Emden function. Density is rho = rho_c theta^n
#                [UMD], pressure P = P_c theta^(n+1) (from P = K rho^(1+1/n)
#                with rho = rho_c theta^n -- general knowledge algebra), and
#                temperature T = T_c theta (Part 4).
#   q(x)         the enclosed mass fraction m(r)/M.
#
# WHERE q COMES FROM
# [WVU] gives the total mass as M = 4 pi alpha^3 rho_c (-xi_1^2 theta'(xi_1)).
# The same derivation applied out to any radius xi (not just the surface)
# gives the enclosed mass m(xi) = 4 pi alpha^3 rho_c (-xi^2 theta'(xi)):
# the derivation integrates 4 pi r^2 rho dr from 0 to r and uses the
# Lane-Emden equation to do the integral, and nothing in it requires the
# upper limit to be the surface (general knowledge; Chandrasekhar Ch. IV).
# Dividing the two:
#
#     q = m/M = (xi^2 theta'(xi)) / (xi_1^2 theta'(xi_1))
#
# All the dimensional factors (alpha, rho_c) cancel, so q depends only on n.
# The check in Part 6 confirms this numerically by integrating the density.
# -----------------------------------------------------------------------------
def dimensionless_profiles(n):
    sol, xi_1, dtheta_1 = solve_scipy(n)

    xi = X_GRID * xi_1
    theta = np.empty_like(xi)
    dtheta = np.empty_like(xi)

    # Centre (x = 0): the integration started at XI0, not 0, so use the exact
    # boundary values theta(0) = 1, theta'(0) = 0.
    theta[0], dtheta[0] = 1.0, 0.0
    # Everywhere else: SciPy's dense output.
    vals = sol.sol(xi[1:])
    theta[1:], dtheta[1:] = vals[0], vals[1]

    # Surface (x = 1): theta is zero by definition. The interpolant gives
    # something like 1e-14 or -1e-14 there (round-off); set it exactly.
    theta[-1] = 0.0
    theta = np.maximum(theta, 0.0)

    q = (xi**2 * dtheta) / (xi_1**2 * dtheta_1)
    q[0], q[-1] = 0.0, 1.0       # exact at centre and surface by definition

    return xi_1, dtheta_1, theta, q, sol


# =============================================================================
# PART 3: Scaling constants that depend only on n
# =============================================================================
#
# These turn the dimensionless solution into physical units once the user
# picks M, R, mu. They are the formulas the JavaScript will use.
#
# (a) CENTRAL DENSITY.  [WVU] writes rho_c = D_n * rho_mean, where
#     rho_mean = M / ((4/3) pi R^3). Derivation from [WVU]'s mass formula:
#         M = 4 pi alpha^3 rho_c (-xi_1^2 theta'_1),   alpha = R/xi_1
#      => rho_c = M xi_1^3 / (4 pi R^3 (-xi_1^2 theta'_1))
#               = M / ((4/3) pi R^3) * [ -xi_1 / (3 theta'_1) ]
#      => D_n = rho_c / rho_mean = -xi_1 / (3 theta'(xi_1))
#
# (b) CENTRAL PRESSURE.  [UMD] gives the radius scale
#         alpha^2 = (n+1) P_c / (4 pi G rho_c^2),
#     so P_c = 4 pi G alpha^2 rho_c^2 / (n+1). Substituting alpha = R/xi_1 and
#     rho_c from (a), and simplifying (algebra, general knowledge):
#         P_c = W_n * G M^2 / R^4,     W_n = 1 / (4 pi (n+1) theta'(xi_1)^2)
#     Check at n = 0 (uniform density): theta'_1 = -sqrt(6)/3, so
#     theta'_1^2 = 2/3 and P_c = 3 G M^2 / (8 pi R^4) -- the standard result
#     for a uniform-density sphere (general knowledge), so the formula holds
#     even at n = 0 where the polytropic constant K itself is undefined.
#
# (c) CENTRAL TEMPERATURE.  The ideal-gas law P = rho k T / (mu m_H)
#     [Caltech] at the centre gives T_c = mu m_H P_c / (k rho_c).
#     And since P/rho = (P_c theta^(n+1)) / (rho_c theta^n) = (P_c/rho_c) theta,
#     the temperature profile is simply T = T_c theta.
#     This ignores radiation pressure. For the n = 3 Eddington option, only a
#     fraction beta of the pressure is gas pressure, so T picks up a factor
#     beta -- that's added in the Eddington step later.
#
# Only D_n and W_n are stored in the JSON (with xi_1 and theta'_1); T_c needs
# mu and the physical constants, so the page computes it.
# -----------------------------------------------------------------------------
def scaling_constants(xi_1, dtheta_1, n):
    D_n = -xi_1 / (3.0 * dtheta_1)
    W_n = 1.0 / (4.0 * np.pi * (n + 1.0) * dtheta_1**2)
    return D_n, W_n


# =============================================================================
# PART 4: Rescale to physical units (the reference for the JavaScript)
# =============================================================================
#
# Physical constants come from SciPy's CODATA values [SciPy-const]:
#   G (gravitational constant), k (Boltzmann constant), m_u (atomic mass unit).
# We use the atomic mass unit for "m_H" in the ideal-gas law. The mass of a
# hydrogen atom is 0.8% larger (general knowledge); either is used in
# textbooks, and the choice changes T_c by under 1%. State which you used.
#
# Solar values: IAU 2015 Resolution B3 nominal values [IAU-B3]:
#   R_sun = 6.957e8 m, (GM)_sun = 1.3271244e20 m^3 s^-2, so
#   M_sun = (GM)_sun / G.
# -----------------------------------------------------------------------------
G = const.G
K_B = const.k
M_U = const.m_u
R_SUN = 6.957e8
M_SUN = 1.3271244e20 / G


def physical_profiles(entry, M, R, mu):
    """Physical profiles for one stored n, given M [kg], R [m], mu.

    `entry` is one item of the JSON "models" list (or the same dict before
    saving). Returns a dict of SI arrays on the stored x grid plus the
    central values.
    """
    n = entry["n"]
    x = np.asarray(X_GRID)
    theta = np.asarray(entry["theta"])
    q = np.asarray(entry["q"])

    rho_mean = M / (4.0 / 3.0 * np.pi * R**3)
    rho_c = entry["D_n"] * rho_mean                      # Part 3(a)
    P_c = entry["W_n"] * G * M**2 / R**4                 # Part 3(b)
    T_c = mu * M_U * P_c / (K_B * rho_c)                 # Part 3(c)

    return {
        "r": x * R,
        "rho": rho_c * theta**n,         # [UMD]; note 0**0 = 1, so n = 0 is uniform
        "P": P_c * theta**(n + 1.0),
        "T": T_c * theta,
        "m": q * M,
        "rho_c": rho_c, "P_c": P_c, "T_c": T_c,
    }


# =============================================================================
# PART 5: Build the data set and write polytropes.json
# =============================================================================
#
# JSON is a plain-text format that JavaScript reads natively (general
# knowledge), so the web page can load this file with one fetch() call.
#
# Values are rounded to 8 significant figures to keep the file small. That's
# far more precision than a plot can show, and far below the grid
# interpolation error measured in Part 6.
# -----------------------------------------------------------------------------
def round_sig(arr, sig=8):
    return [float(f"{v:.{sig}g}") for v in np.asarray(arr, dtype=float)]


def build_dataset():
    print(f"Solving Lane-Emden with SciPy for {len(N_VALUES)} values of n ...")
    models = []
    for n in N_VALUES:
        xi_1, dtheta_1, theta, q, _ = dimensionless_profiles(n)
        D_n, W_n = scaling_constants(xi_1, dtheta_1, n)
        models.append({
            "n": float(n),
            "xi_1": float(xi_1),
            "dtheta_1": float(dtheta_1),
            "D_n": float(D_n),
            "W_n": float(W_n),
            "theta": round_sig(theta),
            "q": round_sig(q),
        })

    data = {
        "description": (
            "Dimensionless Lane-Emden solutions for PHYS 435 stellar model web app. "
            "theta and q = m/M are sampled on x = r/R. Physical profiles: "
            "rho = D_n*rho_mean*theta^n, P = W_n*G*M^2/R^4*theta^(n+1), "
            "T = (mu*m_u*P_c/(k*rho_c))*theta, m = q*M."
        ),
        "solver": "scipy.integrate.solve_ivp, DOP853, rtol=atol=1e-12, series start at xi0=1e-3",
        "x": round_sig(X_GRID),
        "models": models,
    }
    path = os.path.join(OUT_DIR, "polytropes.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  wrote {path}  ({os.path.getsize(path) / 1024:.0f} KB)")

    # Same data as a JavaScript file. Browsers refuse fetch() of a local
    # JSON file when a page is opened by double-clicking (file://), but they
    # will load a <script src="..."> from the same folder. So the web page
    # loads polytropes.js, which works both locally and on GitHub Pages.
    # It just assigns the data to a global variable, POLYTROPE_DATA.
    js_path = os.path.join(OUT_DIR, "polytropes.js")
    with open(js_path, "w") as f:
        f.write("// Generated by generate_polytrope_data.py -- do not edit by hand.\n")
        f.write("globalThis.POLYTROPE_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"  wrote {js_path}")
    return data


# =============================================================================
# PART 6: Checks
# =============================================================================
#
# Each check tests something different, and none of them reuses the formula
# it's checking:
#
#  (1) AGREEMENT WITH STEP 1: xi_1 from SciPy vs your own RK4. Ties the data
#      the web app uses back to the solver you validated in Step 2.
#
#  (2) TOTAL MASS: integrate the density, M = integral of 4 pi r^2 rho dr
#      (the definition of mass, general knowledge), and compare with the M we
#      put in. This independently tests D_n (central density) and the
#      theta^n density profile. Uses a fine grid from SciPy's dense output and
#      the trapezoid rule.
#
#  (3) HYDROSTATIC EQUILIBRIUM: dP/dr = -G m rho / r^2 [UMD]. We compute dP/dr
#      by finite differences of the rescaled P(r) and compare with
#      -G m rho / r^2 built from the rescaled m and rho. This independently
#      tests W_n (central pressure) and q (enclosed mass). It's the physics
#      the whole model is supposed to obey, so it's the strongest check.
#
#  (4) GRID RESOLUTION: the web page will draw straight lines between the 401
#      stored points. Measure the worst error that causes, by comparing the
#      linear interpolation of the stored theta with SciPy's exact dense
#      output halfway between grid points.
#
#  (5) SUN SANITY CHECK: central values for M_sun, R_sun at n = 1.5 and n = 3,
#      with mu = 0.62 -- a typical value for the Sun's ionized interior
#      (general knowledge; find a textbook value to cite when you use it).
#      These are NOT validation (the Sun isn't a polytrope, as the prof
#      said); they are a check that the units come out sensibly, and they
#      become your Step 3 comparison with the Standard Solar Model.
# -----------------------------------------------------------------------------
def run_checks(data):
    models = {m["n"]: m for m in data["models"]}
    test_ns = [0.0, 1.0, 1.5, 3.0, 4.5]

    print("\nCheck 1: SciPy xi_1 (in the JSON) vs your own RK4 from Step 1")
    print("   n      xi_1 SciPy        xi_1 own RK4      |difference|")
    for n in test_ns:
        own = solve_lane_emden(n)[3]
        print(f"   {n:<5}  {models[n]['xi_1']:.10f}   {own:.10f}   {abs(own - models[n]['xi_1']):.1e}")

    print("\nCheck 2: total mass from integrating 4 pi r^2 rho dr  (should be 1)")
    print("Check 3: hydrostatic equilibrium, max relative residual of")
    print("         dP/dr + G m rho / r^2  (should be small)")
    print("   n      M_integrated / M_input     HSE residual")
    M, R, mu = M_SUN, R_SUN, 0.62
    for n in test_ns:
        # Build a FINE version of this model straight from SciPy, so these
        # checks measure the physics/scaling, not the storage grid.
        xi_1, dtheta_1, _, _, sol = dimensionless_profiles(n)
        D_n, W_n = scaling_constants(xi_1, dtheta_1, n)
        x_fine = np.linspace(0, 1, 20001)
        xi = x_fine * xi_1
        th = np.ones_like(xi); dth = np.zeros_like(xi)
        th[1:], dth[1:] = sol.sol(xi[1:])
        th = np.maximum(th, 0.0); th[-1] = 0.0
        q = xi**2 * dth / (xi_1**2 * dtheta_1); q[0] = 0.0
        fine = {"n": n, "D_n": D_n, "W_n": W_n, "theta": th, "q": q}

        # Temporarily point the rescaling at the fine grid.
        global X_GRID
        saved, X_GRID = X_GRID, x_fine
        p = physical_profiles(fine, M, R, mu)
        X_GRID = saved

        # np.trapezoid is NumPy >= 2.0; older versions call it np.trapz.
        trap = getattr(np, "trapezoid", None) or np.trapz
        mass_ratio = trap(4 * np.pi * p["r"]**2 * p["rho"], p["r"]) / M

        # HSE on the interior (skip the centre, where r = 0, and the very
        # edge, where finite differences are one-sided).
        dPdr = np.gradient(p["P"], p["r"])
        rhs = -G * p["m"][1:-1] * p["rho"][1:-1] / p["r"][1:-1]**2
        resid = np.abs(dPdr[1:-1] - rhs).max() / np.abs(rhs).max()
        print(f"   {n:<5}  {mass_ratio:.8f}                {resid:.1e}")

    print("\nCheck 4: storage-grid resolution (401 points), worst error in theta")
    print("         from straight-line interpolation between stored points")
    worst = (0, None)
    for n in N_VALUES:
        _, _, _, _, sol = dimensionless_profiles(n)
        m = models[float(n)]
        x_mid = 0.5 * (X_GRID[1:] + X_GRID[:-1])
        true = np.maximum(sol.sol(x_mid * m["xi_1"])[0], 0.0)
        interp = np.interp(x_mid, data["x"], m["theta"])
        err = np.abs(interp - true).max()
        if err > worst[0]:
            worst = (err, float(n))
    print(f"   worst case: |error| = {worst[0]:.1e} at n = {worst[1]}"
          f"   (theta runs from 0 to 1, so this is a fraction of the full range)")

    print("\nCheck 5: the Sun as a polytrope (M_sun, R_sun, mu = 0.62)")
    print("   n      rho_c/rho_mean    rho_c [kg/m^3]   P_c [Pa]      T_c [K]")
    for n in [1.5, 3.0]:
        p = physical_profiles(models[n], M_SUN, R_SUN, 0.62)
        print(f"   {n:<5}  {models[n]['D_n']:10.3f}       {p['rho_c']:10.4g}     "
              f"{p['P_c']:.3e}    {p['T_c']:.3e}")


if __name__ == "__main__":
    dataset = build_dataset()
    run_checks(dataset)
    print("\nDone.")
