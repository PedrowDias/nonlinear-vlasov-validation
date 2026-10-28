"""Fig. 6 and Table I: the LHC resistive wall, a thick copper pipe of 1 cm radius over the circumference, N_b = 6e11,
tau_b = 1 ns. Solvers over Q' = -10 to +20: DELPHI at (6,24), and the nonlinear solver at (6,36) held against (6,24).
Simulations at Q' = +5 and +15 against the number of slices (400 to 25 600): kicked by 0.1 sigma_x, tracked for 8 e-foldings
of the slower bucket's predicted growth (stopped once the centroid reaches 1 mm), fitted over the last third; 1e6, 2e6 and
4e6 macroparticles and 3 or 10 turns of wake memory as the slicing gets finer (`settings`). Writes data/resistive_wall.json.

    python3 resistive_wall.py               # solvers, then every simulation (days: the finest slicings take hours per run)
    python3 resistive_wall.py 12800 25600   # solvers, then only these slicings (a stored run is not repeated)
    python3 resistive_wall.py solvers       # solvers only, keeping the stored simulations (their predictions updated)
"""
import sys
import numpy as np
import xtrack as xt
import xwakes as xw

import common as c
from common import DELPHI
import nonlinear
from nonlinear import alpha_from_sigma_z, Jz_upper_from_bunch, g0_gaussian, G_I_Cache, compute_impedance_matrix, \
    compute_damper_matrix, computes_coef, eigenmodesDELPHI, classic_thick_wall_formula

circ = 26658.8831999989
E0 = 6.5e12
Qx, Qy = 62.31, 60.32
alpha_p = 3.48e-4
Qs = 1.909e-3
h = 35640
Nb = 6e11
taub = 1e-9
rho, radius = 1.7e-8, 0.01
nemitt = 2.5e-6

Qpscan = np.arange(-10., 20.1, 1.)
lmax, nmax, nmax_check = 6, 24, 36
abseps = 1e-4

slices = {5.: [400, 800, 1600, 3200, 6400, 12800, 25600], 15.: [800, 3200, 6400, 12800, 25600]}
settings = lambda N: (1_000_000, 3) if N <= 3200 else (2_000_000, 3) if N <= 6400 else (4_000_000, 10)   # particles, turns of memory
max_turns = 150_000
efoldings = 8.                  # of the slower bucket
fit_fraction = 3                # fitted over the last third of the run
kick_sigma = 0.1
amp_max = 1e-3
seed = 1

gamma = E0/xt.PROTON_MASS_EV
beta0 = np.sqrt(1 - 1/gamma**2)
v = beta0*c.clight
omega0 = 2*np.pi*v/circ
f0 = omega0/(2*np.pi)
R = circ/(2*np.pi)
eta = alpha_p - 1/gamma**2
omegas = Qs*omega0
Qfrac = Qx - np.floor(Qx)
sigma_z = taub*v/4
beta_x = R/Qx
sigma_x = np.sqrt(nemitt/(beta0*gamma)*beta_x)
V_rf = Qs**2*2*np.pi*beta0**2*E0/(eta*h)
bets = eta*circ/(2*np.pi*Qs)
zeta_half_range = np.pi*R/h
alpha = alpha_from_sigma_z(sigma_z, R, h)

freq = 10.**np.arange(-1, 15.02, 0.02)
nonlinear.circ, nonlinear.radius, nonlinear.rho, nonlinear.beta = circ, radius, rho, beta0
Z = classic_thick_wall_formula(2.*np.pi*freq)


def predict():
    """most unstable dQ/Qs per Q': DELPHI, nonlinear solver"""
    g, a, b = DELPHI.longdistribution_decomp(taub, typelong='Gaussian')
    r = DELPHI.eigenmodesDELPHI_converged_scan(Qpscan, np.array([0]), [0.], [Nb], [omegas], [0.], 1, omega0, Qx, gamma, eta, a, b, taub, g, Z, freq,
                                               kmax=1, kmaxplot=1, crit=5e-2, abseps=abseps, fixed_lmax_nmax=(lmax, nmax))
    Jz_max, kappa_edge = Jz_upper_from_bunch(sigma_z, R, Qs, h, eta)
    g0f = g0_gaussian(alpha, R, Qs, h, eta, kappa_max=kappa_edge)
    cache = G_I_Cache(nmax_check, g0f, R, Qs, h, eta, Jz_upper=Jz_max)
    cd, cz = computes_coef(f0, 0., 1., 0., 1, Nb, gamma, Qx)
    nl = []
    for Qp in Qpscan:
        # two basis sizes, as in the other scans: at Q' = +1 to +5 mode -1 grows at 0.002-0.011 Qs inside the l = -1 band,
        # where eigenvalues that move with nmax are the discretized continuum, not modes
        e = {}
        for n in (nmax, nmax_check):
            matZ, _, cache = compute_impedance_matrix(lmax, n, 0, 1, Qp*omega0/eta, omega0, Qfrac, g0f, Z, freq, R, Qs, h, eta, v,
                                                      Jz_upper=Jz_max, cache=cache, abseps=abseps*Qs*omega0/abs(cz))
            matd = compute_damper_matrix(lmax, n, 0, 1, Qp*omega0/eta, omega0, Qfrac, g0f, R, Qs, h, eta, Jz_upper=Jz_max, cache=cache)
            e[n] = eigenmodesDELPHI(lmax, n, matd, matZ, cd, cz, omegas, R, Qs, h, eta, Jz_upper=Jz_max)[0]/omega0/Qs
        held = c.most_unstable_held(e[nmax_check], e[nmax])
        nl.append(held[0] if held else 0j)       # 0j: no held unstable eigenvalue (growth stored as +0.0 below)
    return {'linear': r['tuneshift_most'][:, 0, 0, 0, 0, 0]/Qs, 'nonlinear': np.array(nl)}


def growth(x):
    """growth rate -Im dQ/Qs of an eigenvalue, +0.0 (never -0.0) when it is not unstable"""
    return -x.imag if -x.imag > 0 else 0.


def one_turn(mode, Qp):
    common = dict(length=circ, qx=Qx, qy=Qy, betx=beta_x, bety=R/Qy, dqx=Qp, dqy=0.)
    if mode == 'linear':
        return xt.LineSegmentMap(longitudinal_mode='linear_fixed_qs', qs=Qs, bets=bets, **common)
    return xt.LineSegmentMap(longitudinal_mode='nonlinear', momentum_compaction_factor=alpha_p,
                             voltage_rf=[V_rf], frequency_rf=[h*f0], phase_rf=[np.pi], **common)


def run(mode, Qp, num_slices, growth_slower):
    """growth rate in Qs over the last third of the run"""
    num_particles, memory = settings(num_slices)
    wake = xw.WakeThickResistiveWall(kind='dipolar_x', radius=radius, length=circ, resistivity=rho)
    wake.configure_for_tracking(zeta_range=(-zeta_half_range, zeta_half_range), num_slices=num_slices, num_turns=memory, circumference=circ)
    line = c.build({'map': one_turn(mode, Qp), 'wake': wake}, p0c=np.sqrt(E0**2 - xt.PROTON_MASS_EV**2))
    zd = c.boltzmann(num_particles, alpha, np.random.default_rng(seed), R=R, h=h, Qs=Qs, eta=eta) if mode == 'nonlinear' else None
    particles = c.matched(line, Nb, num_particles, sigma_z, zd, seed=seed)
    particles.x += kick_sigma*sigma_x
    signal = c.track(line, particles, c.turns(efoldings, growth_slower, max_turns, Qs=Qs), beta_x=beta_x, amp_max=amp_max)
    return c.fit(signal, len(signal) - len(signal)//fit_fraction, Qfrac=Qfrac, Qs=Qs)[0], len(signal)


if __name__ == '__main__':
    only = [int(a) for a in sys.argv[1:] if a != 'solvers']
    p = predict()
    try:
        tracking = c.load('resistive_wall')['tracking']
    except FileNotFoundError:
        tracking = []
    for r in tracking:                       # the stored runs keep their g; their prediction follows the solvers
        r['gp'] = growth(p[r['bucket']][list(Qpscan).index(r['Qp'])])
    out = dict(solvers=dict(Qp=Qpscan.tolist(), linear=[[x.real, growth(x)] for x in p['linear']],      # tolist: json cannot take the array
                            nonlinear=[[x.real, growth(x)] for x in p['nonlinear']]), tracking=tracking)
    c.save('resistive_wall', out)
    if 'solvers' in sys.argv[1:]:
        sys.exit()
    for Qp, scan in slices.items():
        gp = {m: growth(p[m][list(Qpscan).index(Qp)]) for m in p}
        for N in scan:
            if only and N not in only:
                continue
            for mode in ('nonlinear', 'linear'):
                if only and any((r['Qp'], r['bucket'], r['slices']) == (Qp, mode, N) for r in tracking):
                    print(f"Q' = {Qp:+.0f}, {N} slices, {mode}: stored", flush=True); continue      # resume an interrupted run
                g, n = run(mode, Qp, N, min(gp.values()))
                tracking[:] = [r for r in tracking if (r['Qp'], r['bucket'], r['slices']) != (Qp, mode, N)]
                tracking.append(dict(Qp=Qp, bucket=mode, slices=N, particles=settings(N)[0], memory=settings(N)[1], g=g, gp=gp[mode], turns=n))
                print(f"Q' = {Qp:+.0f}, {N} slices, {mode}: tracked/solver {g/gp[mode]:.4f}", flush=True)
                c.save('resistive_wall', out)
