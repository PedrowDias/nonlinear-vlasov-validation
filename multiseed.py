"""Sec. V B: the run-to-run scatter of the simulations. Kicked runs at Q' = +-5, N_b = 3e11, both buckets, with five
different seeds of the macroparticle draw (20 runs, chromaticity.py's machine; 12 e-foldings, stopped once the centroid
reaches 1 mm, fitted over the last quarter but at least the last 3000 turns). Writes data/validation/multiseed.json.

    python3 multiseed.py
"""
import os, json
import numpy as np
import xwakes as xw

import common as c
from nonlinear import computes_coef

Nb = 3e11
Qpscan = [5., -5.]
seeds = [1, 2, 3, 4, 5]
lmax, nmax = 6, 32
abseps = 1e-4
num_particles, num_slices, max_turns, efoldings, kick_sigma, amp_max, fit_turns = 100_000, 300, 60_000, 12, 1e-3, 1e-3, 3000
OUT = os.path.join(c.DATA, 'validation')


def predict():
    b = c.gaussian_bucket(nmax)
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
    nl = [c.eigen(b, lmax, nmax, *c.matrices(b, lmax, nmax, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz)[0] for Qp in Qpscan]
    return {'linear': c.delphi_most(Qpscan, [0.], Nb, c.taub, lmax, nmax)[:, 0],
            'nonlinear': np.array([e[np.argmin(e.imag)] for e in nl])}


def run(mode, Qp, growth_predicted, seed):
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=num_slices)
    line = c.build({'map': c.one_turn(mode, Qp), 'wake': wake})
    zd = c.boltzmann(num_particles, c.alpha, np.random.default_rng(seed)) if mode == 'nonlinear' else None
    particles = c.matched(line, Nb, num_particles, c.sigma_z, zd, seed=seed)     # seeds xpart's draw too
    particles.x += kick_sigma*c.sigma_x
    signal = c.track(line, particles, c.turns(efoldings, growth_predicted, max_turns), amp_max=amp_max)
    i1 = len(signal)
    return c.fit(signal, max(i1//3, i1 - max(fit_turns, i1//4)))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    predicted = predict()
    rows = []
    for i, Qp in enumerate(Qpscan):
        for mode in ('linear', 'nonlinear'):
            p = predicted[mode][i]
            for s in seeds:
                g, re, _ = run(mode, Qp, -p.imag, s)
                rows.append(dict(Qp=Qp, bucket=mode, seed=s, g=g, re=re, gp=-p.imag, rep=p.real))
                print(f"Q' = {Qp:+.0f}, {mode}, seed {s}: growth {g:.5f} ({-p.imag:.5f}, {100*(g/-p.imag - 1):+.2f}%), Re {re:+.4f}", flush=True)
                with open(os.path.join(OUT, 'multiseed.json'), 'w') as f:
                    json.dump(rows, f, indent=1, default=lambda o: o.item())
            dev = np.array([100*(r['g']/r['gp'] - 1) for r in rows if r['Qp'] == Qp and r['bucket'] == mode])
            print(f"==> Q' = {Qp:+.0f}, {mode}: tracked/solver - 1 = {dev.mean():+.2f}% mean, {dev.std(ddof=1):.2f}% rms over {len(dev)} seeds", flush=True)
