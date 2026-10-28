"""Secs. V B and V C: where in the kicked LHC scans does the reference solver have a second mode close to the most unstable
one? A kick excites every mode, and the slope fit gives the most unstable mode's rate only if that mode dominates the signal by
the end of the run; where a second mode grows within a few per cent of it, no run of reasonable length separates them, and
such points are seeded instead (bunch_length.py at 1.6-1.8 ns). For the kicked points of the bunch-length scan (the 18 of the
linearized bucket, the three seeded ones by their kicked runs, g_kicked, and the sinusoidal ones whose kicked run deviates by
more than 1%) and of the resistive-wall scan (Q' = +5, +15, both buckets, at the finest slicing), the reference solver's
fastest modes: DELPHI's six most unstable eigenvalues at the scan's (lmax, nmax) in the linearized bucket, the nonlinear
solver's held eigenvalues at the scan's two basis sizes in the sinusoidal one. A point is flagged when the second mode grows
within 10% of the first.
Writes data/validation/near_degenerate.json.

    python3 near_degenerate.py [bunch_length|resistive_wall ...]      # default: both
"""
import os, sys, json, time
import numpy as np

import common as c
from common import DELPHI
from nonlinear import computes_coef, Jz_upper_from_bunch, g0_gaussian, G_I_Cache, compute_impedance_matrix, compute_damper_matrix, \
    eigenmodesDELPHI

close, kmaxplot = 0.10, 6
OUT = os.path.join(c.DATA, 'validation')
SCANS = ('bunch_length', 'resistive_wall')


def delphi_top(Qpscan, dmax, Nb, taub, lmax, nmax, abseps=1e-4):
    """DELPHI's kmaxplot most unstable dQ/Qs per (Q', damper rate), fastest first (same call as common.delphi_most)"""
    g, a, b = DELPHI.longdistribution_decomp(taub, typelong='Gaussian')
    r = DELPHI.eigenmodesDELPHI_converged_scan(np.atleast_1d(Qpscan), np.array([0]), np.atleast_1d(dmax), [Nb], [c.omegas], [0.], 1,
                                               c.omega0, c.Qx, c.gamma, c.eta, a, b, taub, g, c.Z, c.freq, kmax=1, kmaxplot=kmaxplot,
                                               crit=5e-2, abseps=abseps, fixed_lmax_nmax=(lmax, nmax))
    return r['tuneshift_most'][:, :, 0, 0, 0, :]/c.Qs


def nonlinear_top(b, lmax, nmax, nmax_check, Qp, Nb, abseps=1e-4):
    """the nonlinear solver's held unstable eigenvalues (nmax_check against nmax), fastest first"""
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
    e = {n: c.eigen(b, lmax, n, *c.matrices(b, lmax, n, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz)[0] for n in (nmax, nmax_check)}
    return np.array(c.most_unstable_held(e[nmax_check], e[nmax])[:kmaxplot])


def row(file, keys, bucket, q, modes):
    modes = [m for m in modes if m.imag < 0]
    g = [-m.imag for m in modes]
    second = g[1]/g[0] if len(g) > 1 and g[0] > 0 else None
    return dict(file=file, bucket=bucket, **keys, g=q.get('g'), gp=q['gp'], re=q.get('re'), rep=q.get('rep'),
                dev=100*(q['g']/q['gp'] - 1) if q.get('g') and q.get('gp') else None,
                modes=[[m.real, -m.imag] for m in modes], second_over_first=second, close=bool(second and second > 1 - close))


def run_bunch_length():
    import bunch_length as bl
    rows = c.load('bunch_length')                                              # includes the seeded rows: their kicked values are checked
    out = []
    for taub in bl.taub_scan:
        top = delphi_top(bl.Qpscan, [0.], bl.Nb, taub, bl.lmax, bl.nmax, bl.abseps)[:, 0]
        for q in rows:
            if q['bucket'] == 'linear' and round(q['taub'], 2) == round(taub*1e9, 2) and q.get('g'):
                qq = dict(q, g=q.get('g_kicked', q['g']), re=q.get('re_kicked', q['re']))
                out.append(row('bunch_length', dict(Qp=q['Qp'], taub=q['taub']), 'linear', qq, top[bl.Qpscan.index(q['Qp'])]))
    for q in rows:
        if q['bucket'] == 'nonlinear' and q.get('g') and q.get('gp') and abs(q['g']/q['gp'] - 1) > 0.01:
            b = c.gaussian_bucket(bl.nmax_check, sigma_z=q['taub']*1e-9*c.v/4)
            out.append(row('bunch_length', dict(Qp=q['Qp'], taub=q['taub']), 'nonlinear', q,
                           nonlinear_top(b, bl.lmax, bl.nmax, bl.nmax_check, q['Qp'], bl.Nb, bl.abseps)))
    return out


def run_resistive_wall():
    import resistive_wall as rw
    d = c.load('resistive_wall')
    Qps = sorted(rw.slices)
    finest = {(r['Qp'], r['bucket']): r for r in d['tracking'] if r['slices'] == max(rw.slices[r['Qp']])}
    # linearized bucket: DELPHI at (6,24), as resistive_wall.predict with kmaxplot modes
    g, a, b = DELPHI.longdistribution_decomp(rw.taub, typelong='Gaussian')
    r = DELPHI.eigenmodesDELPHI_converged_scan(np.array(Qps), np.array([0]), [0.], [rw.Nb], [rw.omegas], [0.], 1, rw.omega0, rw.Qx, rw.gamma,
                                               rw.eta, a, b, rw.taub, g, rw.Z, rw.freq, kmax=1, kmaxplot=kmaxplot, crit=5e-2,
                                               abseps=rw.abseps, fixed_lmax_nmax=(rw.lmax, rw.nmax))
    top = r['tuneshift_most'][:, 0, 0, 0, 0, :]/rw.Qs
    # sinusoidal bucket: the nonlinear solver at (6,36) held against (6,24), as resistive_wall.predict
    Jz_max, kappa_edge = Jz_upper_from_bunch(rw.sigma_z, rw.R, rw.Qs, rw.h, rw.eta)
    g0f = g0_gaussian(rw.alpha, rw.R, rw.Qs, rw.h, rw.eta, kappa_max=kappa_edge)
    cache = G_I_Cache(rw.nmax_check, g0f, rw.R, rw.Qs, rw.h, rw.eta, Jz_upper=Jz_max)
    cd, cz = computes_coef(rw.f0, 0., 1., 0., 1, rw.Nb, rw.gamma, rw.Qx)
    out = []
    for i, Qp in enumerate(Qps):
        e = {}
        for n in (rw.nmax, rw.nmax_check):
            matZ, _, cache = compute_impedance_matrix(rw.lmax, n, 0, 1, Qp*rw.omega0/rw.eta, rw.omega0, rw.Qfrac, g0f, rw.Z, rw.freq, rw.R, rw.Qs,
                                                      rw.h, rw.eta, rw.v, Jz_upper=Jz_max, cache=cache, abseps=rw.abseps*rw.Qs*rw.omega0/abs(cz))
            matd = compute_damper_matrix(rw.lmax, n, 0, 1, Qp*rw.omega0/rw.eta, rw.omega0, rw.Qfrac, g0f, rw.R, rw.Qs, rw.h, rw.eta,
                                         Jz_upper=Jz_max, cache=cache)
            e[n] = eigenmodesDELPHI(rw.lmax, n, matd, matZ, cd, cz, rw.omegas, rw.R, rw.Qs, rw.h, rw.eta, Jz_upper=Jz_max)[0]/rw.omega0/rw.Qs
        nl = np.array(c.most_unstable_held(e[rw.nmax_check], e[rw.nmax])[:kmaxplot])
        keys = dict(Qp=Qp, slices=max(rw.slices[Qp]))
        out.append(row('resistive_wall', keys, 'linear', finest[(Qp, 'linear')], top[i]))
        out.append(row('resistive_wall', keys, 'nonlinear', finest[(Qp, 'nonlinear')], nl))
    return out


def key(r):
    return r['file'] + '|' + r['bucket'] + '|' + json.dumps({k: r[k] for k in r if k in ('Nb', 'Qp', 'taub', 'damping')}, sort_keys=True)


def point(r):
    return {k: r[k] for k in r if k in ('Nb', 'Qp', 'taub', 'damping', 'slices')}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = [a for a in sys.argv[1:] if a in SCANS] or list(SCANS)
    path = os.path.join(OUT, 'near_degenerate.json')
    out = {}
    for name in which:
        t0 = time.time()
        for r in globals()['run_' + name]():
            out[key(r)] = r
            flag = f"  <-- second mode within {100*close:.0f}%" if r['close'] else ''
            dev = '-' if r['dev'] is None else f"{r['dev']:+.2f}%"
            print(f"{name:>14} {r['bucket']:>9} {point(r)}: tracked {dev}, modes "
                  + ', '.join(f"{m[0]:+.3f}/{m[1]:.5f}" for m in r['modes'][:3]) + (f", second/first {r['second_over_first']:.3f}" if r['second_over_first'] else '') + flag, flush=True)
        stored = {key(r): r for r in (json.load(open(path)) if os.path.exists(path) else []) if r['file'] in SCANS}      # merge: the scans may run in parallel
        stored.update(out); out = stored
        with open(path, 'w') as f:
            json.dump(list(out.values()), f, indent=1, default=lambda o: o.item())
        print(f"{name}: {(time.time() - t0)/60:.1f} min", flush=True)
    flagged = [r for r in out.values() if r['close']]
    print(f"\n{len(flagged)} kicked points with a second mode within {100*close:.0f}% of the most unstable one:")
    for r in flagged:
        dev = '-' if r['dev'] is None else f"{r['dev']:+.2f}%"
        print(f"  {r['file']} {r['bucket']} {point(r)}: second/first {r['second_over_first']:.3f}, tracked {dev}")
    print('done')
