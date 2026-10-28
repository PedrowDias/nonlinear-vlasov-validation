"""Fig. 5 and Table I: LHC head-tail modes at N_b = 3e11, Q' = +-5, with a bunch-by-bunch resistive damper whose damping rate
of a rigid oscillation goes from 0 to 0.3 Q_s. Solvers: DELPHI at (6,32); the nonlinear solver at (6,48), its most unstable
eigenvalue kept if (6,32) reproduces it. Simulations in both buckets, each seeded with the eigenvector of the predicted mode
for 10 e-foldings (seeded.py); where no mode is predicted to grow faster than 1e-3 Q_s (the sinusoidal bucket at Q' = -5 above
0.15 Q_s) the bunch is kicked and tracked for 1e5 turns. Writes data/damper.json.

    python3 damper.py                   # solvers, then the 52 simulations (about 6 h on an Apple M3)
    python3 damper.py solvers           # recompute the solver predictions only, keeping the stored simulations
"""
import sys
import numpy as np
import xwakes as xw
from xfields import TransverseDamper

import common as c
import seeded
from nonlinear import computes_coef

Nb = 3e11
Qpscan = [-5., 5.]
damping = np.round(np.arange(0., 0.3001, 0.025), 3)       # rigid-mode damping rate, in Qs
dmax = damping*2*np.pi*c.Qs                               # per turn
lmax, nmax, nmax_check = 6, 32, 48
abseps = 1e-4

num_particles = 100_000
num_slices = 300
max_turns = 150_000             # of a seeded growing mode (seeded.py)
kick_sigma = 1e-3               # of a kicked run
seed = 1                        # of the macroparticle draw


def predict():
    """most unstable dQ/Qs per (Q', damping): DELPHI; nonlinear solver at nmax_check, kept if nmax reproduces it (else 0)"""
    out = {'linear': c.delphi_most(Qpscan, dmax, Nb, c.taub, lmax, nmax),
           'nonlinear': np.zeros((len(Qpscan), len(damping)), dtype=complex)}
    b = c.gaussian_bucket(nmax_check)
    cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)[1]
    for iQp, Qp in enumerate(Qpscan):
        mats = {n: c.matrices(b, lmax, n, Qp, abseps*c.Qs*c.omega0/abs(cz)) for n in (nmax, nmax_check)}
        for i, d in enumerate(dmax):
            cd, cz = computes_coef(c.f0, d, 1., 0., 1, Nb, c.gamma, c.Qx)
            e = {n: c.eigen(b, lmax, n, *mats[n], cd, cz)[0] for n in mats}
            held = c.most_unstable_held(e[nmax_check], e[nmax])
            if held:
                out['nonlinear'][iQp, i] = held[0]
    return out


def machine(mode, Qp, d):
    """the one-turn map, the wake and the damper (d: damping rate per turn), and the matched bunch. Xsuite's damper subtracts
    gain_x <px> every turn, which damps a rigid oscillation at gain_x/2 per turn"""
    zr = (-c.zeta_half_range, c.zeta_half_range)
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=zr, num_slices=num_slices)
    damper = TransverseDamper(gain_x=2*d, gain_y=0., zeta_range=zr, num_slices=num_slices)
    line = c.build({'map': c.one_turn(mode, Qp), 'wake': wake, 'damper': damper})
    zd = c.boltzmann(num_particles, c.alpha, np.random.default_rng(seed)) if mode == 'nonlinear' else None
    return line, c.matched(line, Nb, num_particles, c.sigma_z, zd)


_seed_matrices = {}


def mode_seed(mode, Qp, d, gp, rep):
    """the nonlinear solver's eigenvector of the predicted mode at (lmax, nmax_check), in the bucket stretched 200 times for
    the linearized rf (seeded.py); None where the bunch is kicked instead"""
    if not seeded.is_seeded(gp, rep):
        return None
    if (mode, Qp) not in _seed_matrices:
        b = c.gaussian_bucket(nmax_check, h_b=seeded.h_of[mode])
        _seed_matrices[(mode, Qp)] = (b,) + seeded.seed_matrices(b, lmax, nmax_check, Qp, Nb)
    b, matZ, matd = _seed_matrices[(mode, Qp)]
    ev, vec = seeded.nearest_mode(b, lmax, nmax_check, matZ, matd, Nb, d, rep, gp)
    return b, ev, vec, lmax, nmax_check, Qp


def run(mode, Qp, damping_Qs, gp, rep):
    """damping_Qs: the damping rate of a rigid oscillation, in Qs"""
    d = damping_Qs*2*np.pi*c.Qs
    line, particles = machine(mode, Qp, d)
    return seeded.run(line, particles, mode_seed(mode, Qp, d, gp, rep), gp, kick_sigma, max_turns)


if __name__ == '__main__':
    predicted = predict()
    if 'solvers' in sys.argv[1:]:
        rows = c.load('damper')
        for r in rows:
            p = predicted[r['bucket']][Qpscan.index(r['Qp']), int(np.argmin(np.abs(damping - r['damping'])))]
            r.update(gp=(-p.imag if -p.imag > 0 else 0.), rep=p.real if -p.imag > 0 else None)
        c.save('damper', rows)
        sys.exit()
    rows = []
    for iQp, Qp in enumerate(Qpscan):
        for i, dm in enumerate(damping):
            for mode in ('linear', 'nonlinear'):
                q = c.row(predicted[mode][iQp, i], None, None, False, Qp=Qp, damping=dm, bucket=mode)      # gp, rep as stored
                r, signal = run(mode, Qp, q['damping'], q['gp'], q['rep'])
                c.save_signal(f'damper_{Qp:+.0f}_{dm}_{mode}', signal)
                q.update(g=r['g'], re=r['re'], run=r['run'])
                rows.append(q)
                print(f"Q' = {Qp:+.0f}, damping {dm:.3f} Qs, {mode}, {r['run']}: growth {'-' if r['g'] is None else format(r['g'], '.4f')} "
                      f"({q['gp']:.4f}), {r['turns']} turns", flush=True)
                c.save('damper', rows)
