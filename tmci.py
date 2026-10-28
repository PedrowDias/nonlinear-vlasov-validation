"""Fig. 2 and Table I: LHC mode coupling at Q' = 0 against intensity, N_b = (0.25-4.4)e11. Solvers at (4,24): DELPHI, and the
nonlinear solver; the threshold is the intensity at which the growth rate reaches 0.02 Q_s. Simulations in both buckets, each
seeded with the eigenvector of the predicted mode (seeded.py): where the most unstable mode grows faster than 5e-3 Q_s (at every
scan intensity above the threshold) that mode, for 10 e-foldings of its predicted growth (at most 60 000 turns); elsewhere mode 0,
for 24 000 turns, to measure its tune. Writes data/tmci.json.

    python3 tmci.py                     # solvers, then the 62 simulations (about 4 h on an Apple M3)
    python3 tmci.py solvers             # recompute the solver predictions and thresholds only, keeping the stored simulations
"""
import sys
import numpy as np
import xwakes as xw
from scipy.optimize import brentq

import common as c
import seeded
from common import DELPHI
from nonlinear import computes_coef, norm_quadrature, norm

Qp = 0.
Nbscan = np.unique(np.round(np.concatenate([np.arange(0.25, 3.26, 0.25), np.arange(3.3, 3.86, 0.05),
                                              np.arange(3.9, 4.41, 0.1)]), 3))*1e11
lmax, nmax = 4, 24
abseps = 1e-4
unstable_Qs = 5e-3              # growth above which the prediction is the most unstable mode, below it mode 0
threshold_Qs = 0.02

num_particles = 100_000
num_slices = 300
max_turns = 60_000              # of a seeded growing mode (seeded.py)
kick_sigma = 0.05               # of a kicked run (none in this scan: every point has a predicted mode)
seed = 1                        # of the macroparticle draw


def solver(mode):
    """Nb -> (eigenvalues dQ/Qs, eigenvectors (linear) or dipole moment of each eigenvector (sinusoidal)), matrices once"""
    if mode == 'linear':
        g, a, b = DELPHI.longdistribution_decomp(c.taub, typelong='Gaussian')
        matZ = DELPHI.compute_impedance_matrix(lmax, nmax, 0, 1, 0., c.omega0, c.Qfrac, a, b, c.taub, g, c.Z, c.freq, abseps=abseps/10)[0]
        matd = DELPHI.compute_damper_matrix(lmax, nmax, 0, 1, 0., c.omega0, c.Qfrac, a, b, c.taub, g)

        def f(Nb):
            cd, cz = DELPHI.computes_coef(c.f0, 0., b, g[0], 1., c.taub, 0., 1, Nb, c.gamma, c.Qx)
            eig, vec = DELPHI.eigenmodesDELPHI(lmax, nmax, matd, matZ, cd, cz, c.omegas, flageigenvect=True)
            return eig/c.omega0/c.Qs, vec
        return f
    b = c.gaussian_bucket(nmax)
    cz_max = computes_coef(c.f0, 0., 1., 0., 1, Nbscan.max(), c.gamma, c.Qx)[1]
    matZ, matd = c.matrices(b, lmax, nmax, Qp, abseps*c.Qs*c.omega0/abs(cz_max))
    nq = norm_quadrature(lmax, nmax, c.R, c.Qs, c.h, c.eta, b['g0f'], Jz_upper=b['Jz_max'])
    I0 = b['cache'].get_batch(0, np.array([0.]))[1][:nmax+1, 0]          # rigid displacement at Q' = 0

    def f(Nb):
        eig, vec = c.eigen(b, lmax, nmax, matZ, matd, *computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx), vectors=True)
        return eig, np.array([np.abs(vec[lmax, :, k] @ I0)/norm(vec[:, :, k], nq) for k in range(vec.shape[-1])])
    return f


def predict(mode):
    """per Nb, (growth, Re dQ) in Qs of the mode the centroid shows: the most unstable one above unstable_Qs, else
    mode 0 (linearized bucket: the l = 0 eigenvector, followed by overlap; sinusoidal: the largest dipole moment);
    and the threshold"""
    f = solver(mode)
    rows, prev = [], None
    for Nb in Nbscan:
        dQ, x = f(Nb)
        if mode == 'linear':
            w = x.reshape(-1, x.shape[-1])
            w = w/np.linalg.norm(w, axis=0)
            if prev is None:
                w0 = np.sum(np.abs(x[lmax])**2, axis=0)/np.sum(np.abs(x)**2, axis=(0, 1))
                k = int(np.argmax(np.where(w0 > 0.5, np.abs(dQ.real), -1)))
            else:
                k = int(np.argmax(np.abs(prev.conj() @ w)))
            prev = w[:, k]
        else:
            k = int(np.argmax(x))
        most = dQ[int(np.argmin(dQ.imag))]
        p = most if -most.imag > unstable_Qs else dQ[k]
        rows.append((-p.imag if -p.imag > unstable_Qs else 0., p.real))
    threshold = brentq(lambda Nb: -f(Nb)[0].imag.min() - threshold_Qs, 3e11, 4e11, xtol=1e6)
    return rows, threshold


def machine(mode, Nb):
    """the one-turn map and the wake, and the matched bunch"""
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=num_slices)
    line = c.build({'map': c.one_turn(mode, Qp), 'wake': wake})
    zd = c.boltzmann(num_particles, c.alpha, np.random.default_rng(seed)) if mode == 'nonlinear' else None
    return line, c.matched(line, Nb, num_particles, c.sigma_z, zd)


_seed_matrices = {}


def mode_seed(mode, Nb, gp, rep):
    """the nonlinear solver's eigenvector of the predicted mode, at (lmax, nmax), in the bucket stretched 200 times for the
    linearized rf (seeded.py); None where the bunch is kicked instead"""
    if not seeded.is_seeded(gp, rep):
        return None
    if mode not in _seed_matrices:
        b = c.gaussian_bucket(nmax, h_b=seeded.h_of[mode])
        _seed_matrices[mode] = (b,) + seeded.seed_matrices(b, lmax, nmax, Qp, Nbscan.max())
    b, matZ, matd = _seed_matrices[mode]
    ev, vec = seeded.nearest_mode(b, lmax, nmax, matZ, matd, Nb, 0., rep, gp)
    return b, ev, vec, lmax, nmax, Qp


def run(mode, Nb, gp, rep):
    line, particles = machine(mode, Nb)
    return seeded.run(line, particles, mode_seed(mode, Nb, gp, rep), gp, kick_sigma, max_turns)


if __name__ == '__main__':
    predicted, threshold = {}, {}
    for mode in ('linear', 'nonlinear'):
        predicted[mode], threshold[mode] = predict(mode)
    print(f"thresholds: DELPHI {threshold['linear']/1e11:.3f}e11, nonlinear solver {threshold['nonlinear']/1e11:.3f}e11", flush=True)
    if 'solvers' in sys.argv[1:]:
        out = c.load('tmci')
        out['threshold'] = {m: t/1e11 for m, t in threshold.items()}
        for r in out['rows']:
            r['gp'], r['rep'] = predicted[r['bucket']][int(np.argmin(np.abs(Nbscan/1e11 - r['Nb'])))]
        c.save('tmci', out)
        sys.exit()
    out = dict(threshold={m: t/1e11 for m, t in threshold.items()}, rows=[])
    for i, Nb in enumerate(Nbscan):
        for mode in ('linear', 'nonlinear'):
            gp, rep = predicted[mode][i]
            Nb_r = round(Nb/1e11, 3)
            r, signal = run(mode, Nb_r*1e11, gp, rep)
            c.save_signal(f'tmci_{Nb_r}_{mode}', signal)
            out['rows'].append(dict(Nb=Nb_r, bucket=mode, g=r['g'], gp=gp, re=r['re'], rep=rep, run=r['run']))
            print(f"Nb = {Nb_r:.2f}e11, {mode}, {r['run']}: growth {'-' if r['g'] is None else format(r['g'], '.4f')} ({gp:.4f}), "
                  f"Re {r['re']:+.4f} ({rep:+.4f}), {r['turns']} turns", flush=True)
            c.save('tmci', out)
