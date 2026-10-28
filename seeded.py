"""How the single-bunch LHC scans start and fit a run (Sec. V B of the paper).

Seeded: the predicted mode is seeded with its eigenvector (the seed formula of Sec. V B): the complex amplitude x - j beta px of every
macroparticle is offset by the mode's dipole moment at the particle's (J_z, phi), scaled to 0.3 sigma_x at its largest. The
eigenvector is the nonlinear solver's, in the sinusoidal bucket itself or, for the linearized bucket, in the bucket stretched
200 times (its linear limit, Sec. V A). The run lasts 10 e-foldings of the predicted growth (coordinates scaled down by 1e3
whenever the centroid passes 1 mm) and is fitted over its last half; a predicted stable mode (mode 0 below the TMCI
threshold) is run for 24 000 turns to measure its tune.
Kicked: where the solver predicts no growing mode (or a growth below 1e-3 Q_s), the bunch is displaced by a small kick and
tracked for 1e5 turns, fitted over the last quarter.
The validation scripts also fit the seeded runs with a matrix pencil (`pencil`), as a cross-check of the slope fit.
"""
import numpy as np
from scipy.special import ellipk, ellipkinc

import common as c
from nonlinear import computes_coef, Jz_of_kappa, basis_arrays

SEED_SIGMA = 0.3                # largest offset of the seed, in sigma_x
AMP_RESCALE = 1e-3              # m: past this centroid the transverse coordinates are scaled down by 1e3
EFOLDINGS = 10                  # of the predicted growth, for a seeded growing mode
STABLE_TURNS = 24_000           # for a seeded stable mode
KICKED_TURNS = 100_000          # where no growing mode is predicted
FIT_SEEDED = 2                  # a seeded run of n turns is fitted from turn n//FIT_SEEDED on: its last half
FIT_KICKED = 4                  # a kicked run from turn n - n//FIT_KICKED on: its last quarter
GROWTH_FLOOR = 1e-3             # Q_s: below it a predicted "growth" is not seeded
h_of = {'linear': c.h/200., 'nonlinear': c.h}       # harmonic number of the bucket the seed is computed in


def seed_matrices(b, lmax, nmax, Qp, Nb_abseps):
    """the nonlinear solver's matrices for the seed, with the convergence tolerance of the line sum set at intensity Nb_abseps"""
    cz = computes_coef(c.f0, 0., 1., 0., 1, Nb_abseps, c.gamma, c.Qx)[1]
    return c.matrices(b, lmax, nmax, Qp, 1e-4*c.Qs*c.omega0/abs(cz))


def nearest_mode(b, lmax, nmax, matZ, matd, Nb, d, rep, gp):
    """the eigenvalue (and eigenvector) nearest the predicted one, rep - j gp, among the unstable ones when gp > 0;
    d is the damper's damping rate per turn"""
    cd, cz = computes_coef(c.f0, d, 1., 0., 1, Nb, c.gamma, c.Qx)
    e, vec = c.eigen(b, lmax, nmax, matZ, matd, cd, cz, vectors=True)
    cand = np.flatnonzero(e.imag < 0) if gp > 0 else np.arange(len(e))
    k = int(cand[np.argmin(np.abs(e[cand] - (rep - 1j*gp)))])
    return e[k], vec[:, :, k]


def is_seeded(gp, rep):
    """a mode is seeded when the solver predicts one: growing faster than GROWTH_FLOOR, or stable with a tune (gp = 0)"""
    return rep is not None and not (0 < gp <= GROWTH_FLOOR)


def mode_pattern(b, vec, zeta, delta, lmax, nmax, Qp):
    """x - j beta px of the mode on the particles, the seed formula of Sec. V B of the paper"""
    sn = np.sin(b['h']*zeta/(2*c.R))
    cn = b['h']*abs(c.eta)*delta/(2*c.Qs)
    kappa = np.clip(np.hypot(sn, cn), 1e-12, 1 - 1e-12)
    phi = np.pi/2 - np.pi/(2*ellipk(kappa**2))*ellipkinc(np.arctan2(sn/kappa, cn/kappa), kappa**2)
    Jz = Jz_of_kappa(kappa, c.R, c.Qs, b['h'], c.eta)
    w = np.zeros(len(zeta), dtype=complex)
    for l in range(-lmax, lmax+1):
        w += np.exp(-1j*l*phi)*(vec[l+lmax] @ basis_arrays(abs(l), nmax, Jz, b['Jz_max'])[0])
    return w*np.exp(-1j*Qp*c.omega0/(c.eta*c.v)*zeta)


def run(line, particles, seed, gp, kick_sigma, max_turns):
    """track a seeded or a kicked run; seed = (bucket, eigenvalue, eigenvector, lmax, nmax, Q') or None.
    Returns (dict(run, g, re, turns), the centroid signal x - j beta px)"""
    if seed is None:
        particles.x += kick_sigma*c.sigma_x
        signal = c.track(line, particles, KICKED_TURNS, amp_rescale=AMP_RESCALE)
        g, re, grew = c.fit(signal, len(signal) - len(signal)//FIT_KICKED)
        return dict(run='kicked', g=g if grew else None, re=re, turns=len(signal)), signal
    b, ev, vec, lmax, nmax, Qp = seed
    w = mode_pattern(b, vec, particles.zeta, particles.delta, lmax, nmax, Qp)
    w *= SEED_SIGMA*c.sigma_x/np.max(np.abs(w))
    particles.x += np.real(w)
    particles.px += -np.imag(w)/c.beta_x
    n = c.turns(EFOLDINGS, -ev.imag, max_turns) if gp > 0 else STABLE_TURNS
    signal = c.track(line, particles, n, amp_rescale=AMP_RESCALE)
    g, re, grew = c.fit(signal, len(signal)//FIT_SEEDED)
    return dict(run='seeded', g=g if grew else None, re=re, turns=len(signal)), signal


def pencil(signal, re_target=None, tol=0.25, components=6):
    """matrix-pencil fit of a centroid signal, the cross-check of the slope fit: the signal with the betatron line shifted to
    zero, decimated to about 1000 samples, is decomposed into `components` damped exponentials, and the pole carrying the most
    energy over the window is kept, among the poles with Re dQ/Qs within tol of re_target when one is given.
    Returns (Re dQ/Qs, growth/Qs, whether that pole lies within tol of re_target)"""
    d = max(1, len(signal)//1000)
    s = (signal*np.exp(-2j*np.pi*c.Qfrac*np.arange(len(signal))))[::d]
    L = len(s)//3
    Y = np.array([s[i:i+L+1] for i in range(len(s) - L)])
    V = np.linalg.svd(Y, full_matrices=False)[2][:components].T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    A = np.linalg.lstsq(np.vander(z, len(s), increasing=True).T, s, rcond=None)[0]
    lam = np.log(z)/d
    re, g = lam.imag/(2*np.pi*c.Qs), lam.real/(2*np.pi*c.Qs)
    energy = np.array([np.sum(np.abs(A[j]*z[j]**np.arange(len(s)))**2) for j in range(len(z))])
    inside = np.abs(re - re_target) < tol if re_target is not None else np.ones(len(z), bool)
    k = int(np.argmax(np.where(inside, energy, -1.))) if inside.any() else int(np.argmax(energy))
    return re[k], g[k], bool(inside[k]) if re_target is not None else True
