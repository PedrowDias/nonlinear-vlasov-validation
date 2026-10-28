"""Fig. 4 and Table I: LHC head-tail modes at Nb = 3e11, Q' = +-5, against bunch length (tau_b = 0.6-1.8 ns in the 2.5 ns
bucket). Solvers: DELPHI at (6,32); the nonlinear solver at (6,48), its most unstable eigenvalue kept if (6,32) reproduces it.
Simulations: the bunch is kicked and tracked for 25 e-foldings of the predicted growth, fitted over the last quarter. At
Q' = +5 and 1.6-1.8 ns in the linearized bucket, where an l = 0 radial mode grows within 2.4-6.4% of mode -1 and a kick
excites both, the point is instead a run seeded with the eigenvector of mode -1 (10 e-foldings, fitted over the last half,
seeded.py); its kicked run is kept in the row as g_kicked, re_kicked. Writes data/bunch_length.json.

    python3 bunch_length.py             # solvers, then the 36 simulations and the 3 seeded ones (about 10 h on an Apple M3)
    python3 bunch_length.py solvers     # recompute the solver predictions only, keeping the stored simulations
"""
import sys
import numpy as np
import xwakes as xw

import common as c
import seeded
from nonlinear import computes_coef

Nb = 3e11
Qpscan = [5., -5.]
taub_scan = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8])*1e-9
lmax, nmax, nmax_check = 6, 32, 48
abseps = 1e-4

num_particles = 100_000
num_slices = 300                # over the bucket; the linearized bucket's slicer is wider, with the same slice width
max_turns = 200_000             # of a kicked run
efoldings = 25                  # of a kicked run: two modes grow at nearly the same rate for long bunches at Q' = +5
max_turns_seeded = 150_000
kick_sigma = 1e-3
amp_rescale = 1e-3
seed = 1                        # of the macroparticle draw
# the linearized points at Q' = +5 seeded with mode -1; the seeded runs take tau_b as written here (1.6e-9 and the
# 1.6*1e-9 of taub_scan differ in the last bit, and with it the macroparticle draw)
SEEDED = {1.6: 1.6e-9, 1.7: 1.7e-9, 1.8: 1.8e-9}
WINDOWS = {'mode -1': lambda x: abs(x.real + 1) < 0.25, 'l=0 radial': lambda x: abs(x.real) < 0.25}


def predict():
    """most unstable dQ/Qs per (Q', tau_b): DELPHI; nonlinear solver at nmax_check, kept if nmax reproduces it (else 0)"""
    out = {'linear': np.array([c.delphi_most(Qpscan, [0.], Nb, taub, lmax, nmax)[:, 0] for taub in taub_scan]).T,
           'nonlinear': np.zeros((len(Qpscan), len(taub_scan)), dtype=complex)}
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
    for it, taub in enumerate(taub_scan):
        b = c.gaussian_bucket(nmax_check, sigma_z=taub*c.v/4)
        for iQp, Qp in enumerate(Qpscan):
            e = {n: c.eigen(b, lmax, n, *c.matrices(b, lmax, n, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz)[0] for n in (nmax, nmax_check)}
            held = c.most_unstable_held(e[nmax_check], e[nmax])
            if held:
                out['nonlinear'][iQp, it] = held[0]
    return out


def solver_modes(b, Qp, Nb=Nb):
    """the nonlinear solver's most unstable eigenvalue in each of WINDOWS (mode -1, l = 0 radial mode) in bucket b, at
    (lmax, nmax_check): (eigenvalue, eigenvector, whether (lmax, nmax) reproduces it)"""
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
    e, vec = c.eigen(b, lmax, nmax_check, *c.matrices(b, lmax, nmax_check, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz, vectors=True)
    e_check = c.eigen(b, lmax, nmax, *c.matrices(b, lmax, nmax, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz)[0]
    out = {}
    for name, inside in WINDOWS.items():
        ks = [k for k in np.argsort(e.imag) if e[k].imag < 0 and inside(e[k])]
        if ks:
            out[name] = (e[ks[0]], vec[:, :, ks[0]], c.held(e[ks[0]], e_check))
    return out


def machine(mode, Qp, taub):
    """the one-turn map and the wake, and the bunch: Gaussian in the linearized bucket, Boltzmann in the sinusoidal one"""
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    sigma_z = taub*c.v/4
    # The linearized rf is unbounded, so its Gaussian is not cut at the bucket edge: it is sliced out to 6 sigma_z, as DELPHI
    # integrates it (the bucket half length is only 2.8 sigma_z at 1.8 ns). The sinusoidal bucket holds its particles inside
    # +-pi R/h by construction.
    half_range = max(c.zeta_half_range, 6*sigma_z) if mode == 'linear' else c.zeta_half_range
    wake.configure_for_tracking(zeta_range=(-half_range, half_range), num_slices=int(round(num_slices*half_range/c.zeta_half_range)))
    line = c.build({'map': c.one_turn(mode, Qp), 'wake': wake})
    if mode == 'linear':
        particles = c.matched(line, Nb, num_particles, sigma_z, seed=seed)
    else:   # xpart only supplies the transverse coordinates (it cannot match more than 0.118 m rms to this bucket)
        alpha = c.alpha_from_sigma_z(sigma_z, c.R, c.h)
        particles = c.matched(line, Nb, num_particles, min(sigma_z, 0.1), c.boltzmann(num_particles, alpha, np.random.default_rng(seed)), seed=seed)
    return line, particles


def kicked(mode, Qp, taub, growth_predicted):
    """(growth, Re dQ, grew, signal) of a kicked run"""
    line, particles = machine(mode, Qp, taub)
    particles.x += kick_sigma*c.sigma_x
    signal = c.track(line, particles, c.turns(efoldings, growth_predicted, max_turns), amp_rescale=amp_rescale)
    return c.fit(signal, len(signal) - len(signal)//4) + (signal,)


def seeded_run(mode, Qp, taub, b, ev, vec):
    """a run seeded with the solver eigenvector vec (eigenvalue ev, computed in bucket b at (lmax, nmax_check)):
    (dict(run, g, re, turns), signal)"""
    line, particles = machine(mode, Qp, taub)
    return seeded.run(line, particles, (b, ev, vec, lmax, nmax_check, Qp), -ev.imag, kick_sigma, max_turns_seeded)


if __name__ == '__main__':
    predicted = predict()
    if 'solvers' in sys.argv[1:]:
        rows = c.load('bunch_length')
        for r in rows:
            p = predicted[r['bucket']][Qpscan.index(r['Qp']), int(np.argmin(np.abs(taub_scan*1e9 - r['taub'])))]
            r.update(gp=(-p.imag if -p.imag > 0 else 0.), rep=p.real if -p.imag > 0 else None)
        c.save('bunch_length', rows)
        sys.exit()
    rows = []
    for iQp, Qp in enumerate(Qpscan):
        for it, taub in enumerate(taub_scan):
            for mode in ('linear', 'nonlinear'):
                p = predicted[mode][iQp, it]
                g, re, grew, signal = kicked(mode, Qp, taub, max(0., -p.imag))
                q = c.row(p, g, re, grew, Qp=Qp, taub=round(taub*1e9, 2), bucket=mode)
                c.save_signal(f"bunch_length_{Qp:+.0f}_{q['taub']}_{mode}_kicked", signal)
                if mode == 'linear' and Qp == 5. and q['taub'] in SEEDED:
                    tb = SEEDED[q['taub']]
                    b = c.gaussian_bucket(nmax_check, sigma_z=tb*c.v/4, h_b=c.h/200.)      # the linear limit of the nonlinear solver
                    ev, vec, _ = solver_modes(b, Qp)['mode -1']
                    r, signal = seeded_run(mode, Qp, tb, b, ev, vec)
                    c.save_signal(f"bunch_length_{Qp:+.0f}_{q['taub']}_{mode}_seeded", signal)
                    q = dict(q, g=r['g'], re=r['re'], g_kicked=q['g'], re_kicked=q['re'], run='seeded mode -1')
                else:
                    q['run'] = 'kicked'
                rows.append(q)
                print(f"Q' = {Qp:+.0f}, tau_b = {taub*1e9:.1f} ns, {mode}, {q['run']}: growth {'-' if q['g'] is None else format(q['g'], '.4f')} "
                      f"({q['gp']:.4f})", flush=True)
                c.save('bunch_length', rows)
