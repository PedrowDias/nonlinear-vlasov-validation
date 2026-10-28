"""Table II and Table I: LHC head-tail modes at N_b = 3e11, Q' = +-5, for four longitudinal distributions of the rms length of
the 1-ns Gaussian: g0 = (1 - kappa^2/kappa_e^2)^p with p = 0 (waterbag), 1/2 (parabolic line density), 1 (parabolic amplitude),
and g0 = exp(-alpha kappa^2). The reference of the linearized bucket is the nonlinear solver in a bucket stretched 200 times
(its linear limit), with the same g0; that of the sinusoidal bucket the nonlinear solver; both at (6,48), held against (6,32).
Every simulation is seeded with the eigenvector of the predicted mode for 10 e-foldings (seeded.py).
Writes data/distributions.json.

    python3 distributions.py            # solvers, then the 16 simulations (about 3 h on an Apple M3)
    python3 distributions.py solvers    # recompute the solver predictions only, keeping the stored simulations
"""
import sys
import numpy as np
import xwakes as xw
from scipy.optimize import brentq

import common as c
import seeded
from nonlinear import Jz_of_kappa, g0_parabolic, G_I_Cache, computes_coef

Nb = 3e11
Qpscan = [5., -5.]
distributions = [('waterbag', 0.), ('parabolic line', 0.5), ('parabolic amplitude', 1.), ('Gaussian', None)]
h_of = {'linear': c.h/200., 'nonlinear': c.h}
lmax, nmax, nmax_check = 6, 32, 48
abseps = 1e-4

num_particles = 100_000
num_slices = 300
max_turns = 150_000             # of a seeded growing mode (seeded.py)
kick_sigma = 1e-3               # of a kicked run (none in this scan)
seed = 1                        # of the macroparticle draw


def rms_length(p, kappa_edge, h_b):
    """rms z of g0 = (1 - kappa^2/kappa_edge^2)^p, whose line density is (kappa_edge^2 - sin^2 theta)^(p+1/2), theta = h z/2R"""
    theta = np.linspace(-np.arcsin(kappa_edge), np.arcsin(kappa_edge), 20001)
    density = np.clip(kappa_edge**2 - np.sin(theta)**2, 0, None)**(p + 0.5)
    return (2*c.R/h_b)*np.sqrt(np.sum(density*theta**2)/np.sum(density))


def bucket(p, h_b):
    """g0 of rms length sigma_z in the bucket of harmonic h_b, and its G, I cache"""
    if p is None:
        return c.gaussian_bucket(nmax_check, h_b=h_b)
    kappa_edge = brentq(lambda k: rms_length(p, k, h_b) - c.sigma_z, 1e-9, 1 - 1e-12, xtol=1e-15)
    Jz_max = float(Jz_of_kappa(np.array([kappa_edge]), c.R, c.Qs, h_b, c.eta)[0])
    g0f = g0_parabolic(kappa_edge, c.R, c.Qs, h_b, c.eta, p=p)
    return dict(h=h_b, alpha=None, kappa_edge=kappa_edge, Jz_max=Jz_max, g0f=g0f,
                cache=G_I_Cache(nmax_check, g0f, c.R, c.Qs, h_b, c.eta, Jz_upper=Jz_max))


def predict():
    """most unstable dQ/Qs per (distribution, Q') at nmax_check that nmax reproduces"""
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
    out = {m: np.zeros((len(distributions), len(Qpscan)), dtype=complex) for m in h_of}
    for mode, h_b in h_of.items():
        for i, (_, p) in enumerate(distributions):
            b = bucket(p, h_b)
            for iQp, Qp in enumerate(Qpscan):
                e = {n: c.eigen(b, lmax, n, *c.matrices(b, lmax, n, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz)[0] for n in (nmax, nmax_check)}
                out[mode][i, iQp] = c.most_unstable_held(e[nmax_check], e[nmax])[0]
    return out


def longitudinal(mode, p, n, rng):
    """(zeta, delta) from g0 by rejection: kappa^2 = sin^2(h z/2R) + (h eta delta/2Qs)^2 in the sinusoidal bucket,
    its small-amplitude limit (h_b/2R)^2 (z^2 + (bets delta)^2) in the linear one"""
    h_b = h_of[mode]
    b = bucket(p, h_b)
    kappa_edge, alpha = b['kappa_edge'], b['alpha']
    z_max = np.pi*c.R/c.h if p is None else 2*c.R*np.arcsin(kappa_edge)/h_b
    d_max = 2*c.Qs/(c.h*abs(c.eta)) if (mode == 'nonlinear' and p is None) else z_max/abs(c.bets)
    zs, ds = [], []
    while sum(len(z) for z in zs) < n:
        z = rng.uniform(-z_max, z_max, 4*n)
        d = rng.uniform(-d_max, d_max, 4*n)
        if mode == 'nonlinear':
            kappa2 = np.sin(c.h*z/(2*c.R))**2 + (c.h*c.eta*d/(2*c.Qs))**2
        else:
            kappa2 = (h_b/(2*c.R))**2*(z**2 + (c.bets*d)**2)
        weight = np.exp(-alpha*kappa2) if p is None else np.clip(1 - kappa2/kappa_edge**2, 0, None)**p
        keep = (kappa2 < (1 if p is None else kappa_edge**2)) & (rng.uniform(0, 1, 4*n) < weight)
        zs.append(z[keep]); ds.append(d[keep])
    return np.concatenate(zs)[:n], np.concatenate(ds)[:n]


def machine(mode, p, Qp):
    """the one-turn map and the wake, and the bunch: xpart's transverse coordinates, (zeta, delta) drawn from g0"""
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=num_slices)
    line = c.build({'map': c.one_turn(mode, Qp), 'wake': wake})
    return line, c.matched(line, Nb, num_particles, c.sigma_z, longitudinal(mode, p, num_particles, np.random.default_rng(seed)))


def mode_seed(mode, p, Qp, gp, rep):
    """the nonlinear solver's eigenvector of the predicted mode at (lmax, nmax_check), for the same g0, in the bucket
    stretched 200 times for the linearized rf (seeded.py)"""
    if not seeded.is_seeded(gp, rep):
        return None
    b = bucket(p, h_of[mode])
    matZ, matd = seeded.seed_matrices(b, lmax, nmax_check, Qp, Nb)
    ev, vec = seeded.nearest_mode(b, lmax, nmax_check, matZ, matd, Nb, 0., rep, gp)
    return b, ev, vec, lmax, nmax_check, Qp


def run(mode, p, Qp, gp, rep):
    line, particles = machine(mode, p, Qp)
    return seeded.run(line, particles, mode_seed(mode, p, Qp, gp, rep), gp, kick_sigma, max_turns)


if __name__ == '__main__':
    predicted = predict()
    if 'solvers' in sys.argv[1:]:
        rows = c.load('distributions')
        names = [d[0] for d in distributions]
        for r in rows:
            p = predicted[r['bucket']][names.index(r['dist']), Qpscan.index(r['Qp'])]
            r.update(gp=(-p.imag if -p.imag > 0 else 0.), rep=p.real if -p.imag > 0 else None)
        c.save('distributions', rows)
        sys.exit()
    rows = []
    for iQp, Qp in enumerate(Qpscan):
        for i, (name, p) in enumerate(distributions):
            for mode in ('linear', 'nonlinear'):
                q = c.row(predicted[mode][i, iQp], None, None, False, Qp=Qp, dist=name, bucket=mode)      # gp, rep as stored
                r, signal = run(mode, p, Qp, q['gp'], q['rep'])
                c.save_signal(f"distributions_{Qp:+.0f}_{name.replace(' ', '-')}_{mode}", signal)
                q.update(g=r['g'], re=r['re'], run=r['run'])
                rows.append(q)
                print(f"Q' = {Qp:+.0f}, {name}, {mode}, {r['run']}: growth {'-' if r['g'] is None else format(r['g'], '.4f')} ({q['gp']:.4f}), "
                      f"{r['turns']} turns", flush=True)
                c.save('distributions', rows)
