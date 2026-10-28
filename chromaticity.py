"""Fig. 3 and Table I: LHC head-tail modes at N_b = 3e11 against chromaticity, Q' = -10 to +10. Solvers at (6,32): DELPHI,
and the nonlinear solver. Simulations in both buckets, each seeded with the eigenvector of the predicted mode (taken at
(6,48)) for 10 e-foldings of its predicted growth (seeded.py); at Q' = 0, where neither solver predicts growth above 1e-3 Q_s,
the bunch is kicked and tracked for 1e5 turns. Writes data/chromaticity.json.

    python3 chromaticity.py             # solvers, then the 42 simulations (about 6 h on an Apple M3)
    python3 chromaticity.py solvers     # recompute the solver predictions only, keeping the stored simulations
"""
import sys
import numpy as np
import xwakes as xw

import common as c
import seeded
from nonlinear import computes_coef

Nb = 3e11
Qpscan = np.arange(-10., 10.1, 1.)
lmax, nmax = 6, 32
nmax_seed = 48                  # basis size of the seed's eigenvector
abseps = 1e-4

num_particles = 100_000
num_slices = 300
max_turns = 150_000             # of a seeded growing mode (seeded.py)
kick_sigma = 1e-3               # of a kicked run
seed = 1                        # of the macroparticle draw


def predict():
    """most unstable dQ/Qs per Q': DELPHI, nonlinear solver"""
    b = c.gaussian_bucket(nmax)
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
    nl = [c.eigen(b, lmax, nmax, *c.matrices(b, lmax, nmax, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz)[0] for Qp in Qpscan]
    return {'linear': c.delphi_most(Qpscan, [0.], Nb, c.taub, lmax, nmax)[:, 0],
            'nonlinear': np.array([e[np.argmin(e.imag)] for e in nl])}


def machine(mode, Qp):
    """the one-turn map and the wake, and the matched bunch"""
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=num_slices)
    line = c.build({'map': c.one_turn(mode, Qp), 'wake': wake})
    zd = c.boltzmann(num_particles, c.alpha, np.random.default_rng(seed)) if mode == 'nonlinear' else None
    return line, c.matched(line, Nb, num_particles, c.sigma_z, zd)


def mode_seed(mode, Qp, gp, rep):
    """the nonlinear solver's eigenvector of the predicted mode at (lmax, nmax_seed), in the bucket stretched 200 times for
    the linearized rf (seeded.py); None where the bunch is kicked instead"""
    if not seeded.is_seeded(gp, rep):
        return None
    b = c.gaussian_bucket(nmax_seed, h_b=seeded.h_of[mode])
    matZ, matd = seeded.seed_matrices(b, lmax, nmax_seed, Qp, Nb)
    ev, vec = seeded.nearest_mode(b, lmax, nmax_seed, matZ, matd, Nb, 0., rep, gp)
    return b, ev, vec, lmax, nmax_seed, Qp


def run(mode, Qp, gp, rep):
    line, particles = machine(mode, Qp)
    return seeded.run(line, particles, mode_seed(mode, Qp, gp, rep), gp, kick_sigma, max_turns)


if __name__ == '__main__':
    predicted = predict()
    if 'solvers' in sys.argv[1:]:
        rows = c.load('chromaticity')
        for r in rows:
            p = predicted[r['bucket']][int(np.argmin(np.abs(Qpscan - r['Qp'])))]
            r.update(gp=(-p.imag if -p.imag > 0 else 0.), rep=p.real if -p.imag > 0 else None)
        c.save('chromaticity', rows)
        sys.exit()
    rows = []
    for i, Qp in enumerate(Qpscan):
        for mode in ('linear', 'nonlinear'):
            p = predicted[mode][i]
            q = c.row(p, None, None, False, Qp=Qp, bucket=mode)                  # gp, rep as stored
            r, signal = run(mode, Qp, q['gp'], q['rep'])
            c.save_signal(f'chromaticity_{Qp:+.0f}_{mode}', signal)
            q.update(g=r['g'], re=r['re'], run=r['run'])
            rows.append(q)
            print(f"Q' = {Qp:+.0f}, {mode}, {r['run']}: growth {'-' if r['g'] is None else format(r['g'], '.4f')} ({q['gp']:.4f}), "
                  f"Re {r['re']:+.4f}, {r['turns']} turns", flush=True)
            c.save('chromaticity', rows)
