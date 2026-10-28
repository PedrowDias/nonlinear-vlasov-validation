"""Damper paragraph of Sec. V C: at Q' = -5, above 0.15 Q_s, the sinusoidal bucket Landau damps mode +1 and the solver finds
no mode growing faster than 3e-7 Q_s. Kicked runs of 5e5 turns (1e6 at 0.15 Q_s, the last point where the mode grows) at 0.15,
0.175, 0.2, 0.25 and 0.3 Q_s, stopped once the centroid reaches 1 mm and fitted over their last two thirds, resolve any growth
above about 2.5e-4 Q_s; the solver's eigenvalues at (6,32) and (6,48)
(the most unstable held one, and the fastest of any kind) are stored with them. Writes data/validation/damper_long.json.

    python3 damper_long.py
"""
import os, json
import numpy as np
import xwakes as xw
from xfields import TransverseDamper

import common as c
from nonlinear import computes_coef

Nb = 3e11
Qp = -5.
damping = [0.15, 0.175, 0.2, 0.25, 0.3]                       # rigid-mode damping rate, in Qs
turns = {0.15: 1_000_000}
default_turns = 500_000
lmax, nmax, nmax_check = 6, 32, 48
abseps = 1e-4
num_particles, num_slices, kick_sigma, amp_max, seed = 100_000, 300, 1e-3, 1e-3, 1
OUT = os.path.join(c.DATA, 'validation')


def predict():
    """nonlinear solver at two sizes: the most unstable held eigenvalue and the fastest-growing eigenvalue of any kind"""
    b = c.gaussian_bucket(nmax_check)
    cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)[1]
    mats = {n: c.matrices(b, lmax, n, Qp, abseps*c.Qs*c.omega0/abs(cz)) for n in (nmax, nmax_check)}
    out = {}
    for d in damping:
        cd, cz = computes_coef(c.f0, d*2*np.pi*c.Qs, 1., 0., 1, Nb, c.gamma, c.Qx)
        e = {n: c.eigen(b, lmax, n, *mats[n], cd, cz)[0] for n in mats}
        held = c.most_unstable_held(e[nmax_check], e[nmax])
        fastest = e[nmax_check][np.argmin(e[nmax_check].imag)]
        out[d] = dict(held=[held[0].real, -held[0].imag] if held else None, fastest_any=[fastest.real, -fastest.imag])
        print(f"damping {d:.3f} Qs: held {out[d]['held']}, fastest eigenvalue of any kind {out[d]['fastest_any']}", flush=True)
    return out


def run(d, num_turns):
    zr = (-c.zeta_half_range, c.zeta_half_range)
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=zr, num_slices=num_slices)
    damper = TransverseDamper(gain_x=2*d*2*np.pi*c.Qs, gain_y=0., zeta_range=zr, num_slices=num_slices)
    line = c.build({'map': c.one_turn('nonlinear', Qp), 'wake': wake, 'damper': damper})
    zd = c.boltzmann(num_particles, c.alpha, np.random.default_rng(seed))
    particles = c.matched(line, Nb, num_particles, c.sigma_z, zd, seed=seed)
    particles.x += kick_sigma*c.sigma_x
    signal = c.track(line, particles, num_turns, amp_max=amp_max)
    i1 = len(signal)
    g, re, grew = c.fit(signal, i1//3)                                 # slope over the last two thirds
    halves = [np.polyfit(np.arange(a, b), np.log(np.abs(signal[a:b])), 1)[0]/(2*np.pi*c.Qs) for a, b in ((i1//3, 2*i1//3), (2*i1//3, i1))]
    return g, re, grew, halves, i1, float(np.abs(signal[-1])/np.abs(signal[0]))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'damper_long.json')
    out = json.load(open(path)) if os.path.exists(path) else None          # resume an interrupted run
    if not out or out.get('Qp') != Qp or not out.get('solvers'):
        out = dict(Qp=Qp, solvers=predict(), tracking=[])
    done = {r['damping'] for r in out['tracking']}
    for d in damping:
        if d in done:
            print(f"damping {d:.3f} Qs: stored", flush=True); continue
        g, re, grew, halves, n, ratio = run(d, turns.get(d, default_turns))
        bound = 1/(2*np.pi*c.Qs*(n - n//3))                           # one e-folding over the fit window, in Qs
        out['tracking'].append(dict(damping=d, g=g, g_halves=halves, re=re, grew=grew, turns=n, amplitude_ratio=ratio, resolvable_growth=bound))
        print(f"damping {d:.3f} Qs, {n} turns: slope {g:+.6f} Qs (halves {halves[0]:+.6f}, {halves[1]:+.6f}); "
              f"amplitude ratio end/start {ratio:.3g}; growth resolvable above {bound:.1e} Qs", flush=True)
        with open(os.path.join(OUT, 'damper_long.json'), 'w') as f:
            json.dump(out, f, indent=1, default=lambda o: o.item())
