"""Sec. V and V C: how the nonlinear solver's predictions depend on the basis size. The two-size rule (common.held: an
eigenvalue of the larger basis that the smaller one reproduces within 0.003 Q_s in Re and 5% in growth) identifies the discrete
modes, but inside the band of incoherent synchrotron frequencies the discretized continuum has a spacing of the same order.
Every stored prediction of the LHC scans is solved again at a third, larger basis size n3 and compared (for the PSB, the
candidates of psb.predict with the impedance at the betatron lines, the held modes within 12% of the fastest):
  - drift: the eigenvalue of the n3 basis nearest the stored one, |dRe| in Q_s and |dg|/g;
  - held3: the stored eigenvalue passes the two-size rule against n3 (held_chain: against both other sizes);
  - fastest: the most unstable eigenvalue at each size inside the stored mode's Re window (+-0.25 Q_s);
  - converged = False when |dg|/g > 2%, |dRe| > 0.003 Q_s or held3 fails.
The paper quotes three results: mode -1 at N_b = 1e11 and 3e10, Q' = +5 (n_max = 32 to 96), the Q' = 0 point of the
chromaticity scan (an in-band eigenvalue absent at 48 and 64), and the damper scan at Q' = -5 (less than 1% between 32, 48
and 64). Scans: chromaticity, bunch_length, distributions, damper, nb_scan, matching (n3 = 64; 64, 80, 96 for nb_scan),
resistive_wall ((6,64) against (6,36) and (6,24)) and psb ((13,64) against (13,32) and (13,24)). Solver only; the rows are
merged into data/validation/convergence_band.json per scan.

    python3 convergence_band.py                        # all scans (a night on an Apple M3)
    python3 convergence_band.py chromaticity damper    # some scans
    python3 convergence_band.py nb_scan --n3 64 80 96  # basis sizes for the N_b scan
"""
import os, sys, json, time
import numpy as np

import common as c
from nonlinear import computes_coef, G_I_Cache

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'convergence_band.json')
SCANS = ('chromaticity', 'bunch_length', 'distributions', 'damper', 'nb_scan', 'matching', 'resistive_wall', 'psb')
N3_DEFAULT = 64
TOL_RE, TOL_G = 3e-3, 0.02


def nearest(e, x):
    """eigenvalue of e nearest to x, and its distance in Re and relative growth"""
    k = int(np.argmin(np.abs(e - x)))
    return e[k], abs(e[k].real - x.real), abs(e[k].imag - x.imag)/abs(x.imag)


def fastest_in_window(e, x, half=0.25):
    """the most unstable eigenvalue of e with |Re - Re x| < half"""
    w = [y for y in e if y.imag < 0 and abs(y.real - x.real) < half]
    return min(w, key=lambda y: y.imag) if w else None


def audit(stored, eigs, sizes):
    """stored: complex dQ/Qs of the stored prediction (computed at sizes[1], held against sizes[0]; the chromaticity scan stores
    sizes[0]); eigs: {n: eigenvalues}; sizes: (n, n_check, n3...)"""
    n, n3 = sizes[0], sizes[-1]
    near3, dre, dg = nearest(eigs[n3], stored)
    held3 = c.held(stored, eigs[n3])
    chain = held3 and c.held(stored, eigs[n])
    fast = {int(m): fastest_in_window(eigs[m], stored) for m in sizes}
    return dict(stored=[stored.real, -stored.imag], n3=int(n3), nearest_n3=[near3.real, -near3.imag], dRe=float(dre), dg_rel=float(dg),
                held3=bool(held3), held_chain=bool(chain),
                fastest={str(m): ([x.real, -x.imag] if x is not None else None) for m, x in fast.items()},
                converged=bool(held3 and dre <= TOL_RE and dg <= TOL_G))


def lhc_eigs(b, lmax, Qp, Nb, d, sizes, abseps):
    """eigenvalues of the nonlinear solver in bucket b at every size (the bucket's cache must have been built at max(sizes))"""
    cd, cz = computes_coef(c.f0, d, 1., 0., 1, Nb, c.gamma, c.Qx)
    out = {}
    for m in sizes:
        matZ, matd = c.matrices(b, lmax, m, Qp, abseps*c.Qs*c.omega0/abs(cz))
        out[m] = c.eigen(b, lmax, m, matZ, matd, cd, cz)[0]
    return out


def stored_value(q, g_min=1e-4):
    """complex dQ/Qs of a stored row (gp, rep), or None if it predicts no growth (gp below g_min: the damper scan stores
    eigenvalues growing at < 3e-7 Qs where the solver finds no mode, Sec. V C)"""
    if q.get('gp') and q['gp'] >= g_min and q.get('rep') is not None:
        return complex(q['rep'], -q['gp'])
    return None


def log(name, keys, r, t0):
    f = r['fastest']
    print(f"{name:>14} {keys}: stored {r['stored'][0]:+.4f}/{r['stored'][1]:.5f} -> n3={r['n3']} nearest {r['nearest_n3'][0]:+.4f}/{r['nearest_n3'][1]:.5f} "
          f"(dRe {r['dRe']:.4f}, dg {100*r['dg_rel']:.2f}%) held3 {r['held3']} | fastest per size "
          + ', '.join(f"{m}: {('%+.4f/%.5f' % tuple(v)) if v else 'none'}" for m, v in f.items())
          + f" | {'OK' if r['converged'] else 'NOT CONVERGED'} [{(time.time()-t0)/60:.1f} min]", flush=True)


def run_chromaticity(n3):
    import chromaticity as ch
    rows = [q for q in c.load('chromaticity') if q['bucket'] == 'nonlinear']
    sizes = (ch.nmax, 48, n3)                              # chromaticity.py stores the (6,32) value with no second size; 48 is added here
    b = c.gaussian_bucket(n3)
    out = []
    for q in rows:
        x = stored_value(q)
        if x is None:
            continue
        t0 = time.time()
        e = lhc_eigs(b, ch.lmax, q['Qp'], ch.Nb, 0., sizes, ch.abseps)
        r = dict(scan='chromaticity', Qp=q['Qp'], bucket='nonlinear', sizes=list(sizes), **audit(x, e, sizes))
        log('chromaticity', dict(Qp=q['Qp']), r, t0); out.append(r)
    return out


def run_bunch_length(n3):
    import bunch_length as bl
    rows = [q for q in c.load('bunch_length') if q['bucket'] == 'nonlinear']
    sizes = (bl.nmax, bl.nmax_check, n3)
    out = []
    for taub in bl.taub_scan:
        b = c.gaussian_bucket(n3, sigma_z=taub*c.v/4)
        for q in rows:
            if round(q['taub'], 2) != round(taub*1e9, 2):
                continue
            x = stored_value(q)
            if x is None:
                continue
            t0 = time.time()
            e = lhc_eigs(b, bl.lmax, q['Qp'], bl.Nb, 0., sizes, bl.abseps)
            r = dict(scan='bunch_length', Qp=q['Qp'], taub=q['taub'], bucket='nonlinear', sizes=list(sizes), **audit(x, e, sizes))
            log('bunch_length', dict(Qp=q['Qp'], taub=q['taub']), r, t0); out.append(r)
    return out


def run_distributions(n3):
    import distributions as di
    rows = c.load('distributions')
    sizes = (di.nmax, di.nmax_check, n3)
    out = []
    di.nmax_check = n3                                   # di.bucket builds its cache at di.nmax_check
    for mode, h_b in di.h_of.items():
        for name, p in di.distributions:
            b = di.bucket(p, h_b)
            for Qp in di.Qpscan:
                q = [r for r in rows if r.get('bucket') == mode and r.get('dist') == name and r.get('Qp') == Qp]
                x = stored_value(q[0]) if q else None
                if x is None:
                    continue
                t0 = time.time()
                e = lhc_eigs(b, di.lmax, Qp, di.Nb, 0., sizes, di.abseps)
                r = dict(scan='distributions', Qp=Qp, distribution=name, bucket=mode, sizes=list(sizes), **audit(x, e, sizes))
                log('distributions', dict(Qp=Qp, dist=name, bucket=mode), r, t0); out.append(r)
    return out


def run_damper(n3):
    import damper as da
    rows = [q for q in c.load('damper') if q['bucket'] == 'nonlinear']
    sizes = (da.nmax, da.nmax_check, n3)
    b = c.gaussian_bucket(n3)
    out = []
    cz = computes_coef(c.f0, 0., 1., 0., 1, da.Nb, c.gamma, c.Qx)[1]
    for Qp in da.Qpscan:
        mats = {m: c.matrices(b, da.lmax, m, Qp, da.abseps*c.Qs*c.omega0/abs(cz)) for m in sizes}
        for q in rows:
            if q['Qp'] != Qp:
                continue
            x = stored_value(q)
            if x is None:
                continue
            t0 = time.time()
            d = q['damping']*2*np.pi*c.Qs
            cd, cz = computes_coef(c.f0, d, 1., 0., 1, da.Nb, c.gamma, c.Qx)
            e = {m: c.eigen(b, da.lmax, m, *mats[m], cd, cz)[0] for m in sizes}
            r = dict(scan='damper', Qp=Qp, damping=q['damping'], bucket='nonlinear', sizes=list(sizes), **audit(x, e, sizes))
            log('damper', dict(Qp=Qp, damping=q['damping']), r, t0); out.append(r)
    return out


def run_nb_scan(n3_list):
    import nb_scan as nb
    data = json.load(open(os.path.join(OUT, 'nb_scan.json')))
    sizes = (nb.nmax, nb.nmax_check) + tuple(n3_list)
    b = c.gaussian_bucket(max(sizes))
    cz0 = computes_coef(c.f0, 0., 1., 0., 1, 3e11, c.gamma, c.Qx)[1]
    mats = {m: c.matrices(b, nb.lmax, m, nb.Qp, nb.abseps*c.Qs*c.omega0/abs(cz0)) for m in sizes}
    out = []
    for s in data['solvers']:
        Nb = s['Nb']
        t0 = time.time()
        cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
        e = {m: c.eigen(b, nb.lmax, m, *mats[m], cd, cz)[0] for m in sizes}
        for key in ('nonlinear_most', 'nonlinear_mode_m1'):
            if s.get(key) is None:
                continue
            x = complex(s[key][0], -s[key][1])
            r = dict(scan='nb_scan', Nb=Nb, which=key, bucket='nonlinear', sizes=list(sizes), **audit(x, e, sizes))
            # the whole l = -1 window at every size, fastest three, to document the reshuffling
            r['window_m1'] = {str(m): [[y.real, -y.imag] for y in sorted((y for y in e[m] if y.imag < 0 and abs(y.real + 1) < 0.25), key=lambda y: y.imag)[:3]] for m in sizes}
            log('nb_scan', dict(Nb=Nb, which=key), r, t0); out.append(r)
    return out


def run_matching(n3):
    data = json.load(open(os.path.join(OUT, 'matching.json')))      # matching.py is not imported: it runs at import
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, 3e11, c.gamma, c.Qx)
    out = []
    for m in data['matchings']:
        alpha = m['alpha']
        Jz_max, kappa_edge = c.Jz_upper_from_bunch(m['sigma_z_cm']/100, c.R, c.Qs, c.h, c.eta)
        g0f = c.g0_gaussian(alpha, c.R, c.Qs, c.h, c.eta, kappa_max=kappa_edge)
        b = dict(h=c.h, alpha=alpha, kappa_edge=kappa_edge, Jz_max=Jz_max, g0f=g0f, cache=G_I_Cache(n3, g0f, c.R, c.Qs, c.h, c.eta, Jz_upper=Jz_max))
        for Qp in (5., -5.):
            g = m['growth'].get(str(Qp))
            if not g:
                continue
            x = complex(g[0], -g[1])
            t0 = time.time()
            e = lhc_eigs(b, 6, Qp, 3e11, 0., (32, 48, n3), 1e-4)
            r = dict(scan='matching', matching=m['matching'], Qp=Qp, bucket='nonlinear', sizes=[32, 48, n3], **audit(x, e, (32, 48, n3)))
            log('matching', dict(matching=m['matching'], Qp=Qp), r, t0); out.append(r)
    return out


def run_resistive_wall(n3):
    import resistive_wall as rw
    data = c.load('resistive_wall')
    sol = data['solvers']                                     # {'Qp': [...], 'linear': [[Re, g], ...], 'nonlinear': [[Re, g], ...]}
    sizes = (rw.nmax, rw.nmax_check, n3)
    Jz_max, kappa_edge = rw.Jz_upper_from_bunch(rw.sigma_z, rw.R, rw.Qs, rw.h, rw.eta)
    g0f = rw.g0_gaussian(rw.alpha, rw.R, rw.Qs, rw.h, rw.eta, kappa_max=kappa_edge)
    cache = G_I_Cache(n3, g0f, rw.R, rw.Qs, rw.h, rw.eta, Jz_upper=Jz_max)
    cd, cz = computes_coef(rw.f0, 0., 1., 0., 1, rw.Nb, rw.gamma, rw.Qx)
    out = []
    for Qp in rw.Qpscan:
        j = int(np.argmin(np.abs(np.array(sol['Qp']) - Qp)))
        re_s, g_s = sol['nonlinear'][j]
        stored = complex(re_s, -g_s) if g_s > 0 else None     # None: no held unstable eigenvalue was stored
        t0 = time.time()
        e = {}
        for m in sizes:
            matZ, _, cache = rw.compute_impedance_matrix(rw.lmax, m, 0, 1, Qp*rw.omega0/rw.eta, rw.omega0, rw.Qfrac, g0f, rw.Z, rw.freq, rw.R, rw.Qs, rw.h, rw.eta, rw.v,
                                                         Jz_upper=Jz_max, cache=cache, abseps=rw.abseps*rw.Qs*rw.omega0/abs(cz))
            matd = rw.compute_damper_matrix(rw.lmax, m, 0, 1, Qp*rw.omega0/rw.eta, rw.omega0, rw.Qfrac, g0f, rw.R, rw.Qs, rw.h, rw.eta, Jz_upper=Jz_max, cache=cache)
            e[m] = rw.eigenmodesDELPHI(rw.lmax, m, matd, matZ, cd, cz, rw.omegas, rw.R, rw.Qs, rw.h, rw.eta, Jz_upper=Jz_max)[0]/rw.omega0/rw.Qs
        if stored is None:
            held = c.most_unstable_held(e[rw.nmax_check], e[rw.nmax])
            print(f"resistive_wall Qp={Qp:+.0f}: nothing stored; recomputed pair gives {held[0] if held else 'nothing'}", flush=True)
            continue
        r = dict(scan='resistive_wall', Qp=Qp, bucket='nonlinear', sizes=list(sizes), **audit(stored, e, sizes))
        log('resistive_wall', dict(Qp=Qp), r, t0); out.append(r)
    return out


def run_psb(n3):
    import psb
    import nonlinear
    sizes = (psb.nmax, psb.nmax_check, n3)
    out = []
    for mode in ('nonlinear', 'linear'):
        b = psb.bucket(psb.h if mode == 'nonlinear' else psb.h/200.)
        b['cache'] = G_I_Cache(n3, b['g0f'], psb.R, psb.Qs, b['h'], psb.eta, Jz_upper=b['Jz_max'])
        for Qp in psb.Qpscan:
            t0 = time.time()
            nonlinear.Zy_interp = psb.line_sampling
            e = {m: psb.eigenmodes(b, Qp, m)[0] for m in sizes}
            held = c.most_unstable_held(e[psb.nmax_check], e[psb.nmax])
            if not held:
                print(f"psb {mode} Qp={Qp:+.0f}: no held unstable eigenvalue", flush=True)
                continue
            # the candidates of psb.predict: the held modes at the betatron lines within 12% of the fastest, audited at the lines
            fastest = held[0].imag
            for x in held:
                if x.imag > (1 - 1.5*psb.within)*fastest:
                    break
                r = dict(scan='psb', Qp=Qp, bucket=mode, sampling='betatron lines', sizes=list(sizes), **audit(x, e, sizes))
                log('psb', dict(Qp=Qp, bucket=mode, m=round(x.real)), r, t0); out.append(r)
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    n3_list = [N3_DEFAULT]
    if '--n3' in args:
        i = args.index('--n3'); n3_list = [int(a) for a in args[i+1:] if a.isdigit()]; args = args[:i]
    which = [a for a in args if a in SCANS] or list(SCANS)
    os.makedirs(OUT, exist_ok=True)
    stored = json.load(open(PATH)) if os.path.exists(PATH) else []
    for name in which:
        t0 = time.time()
        rows = run_nb_scan(n3_list) if name == 'nb_scan' else globals()['run_' + name](n3_list[-1])
        stored = [r for r in stored if r['scan'] != name] + rows                 # replace this scan's rows
        with open(PATH, 'w') as f:
            json.dump(stored, f, indent=1, default=lambda o: o.item())
        bad = [r for r in rows if not r['converged']]
        print(f"=== {name}: {len(rows)} predictions audited, {len(bad)} not converged at n3 [{(time.time()-t0)/60:.0f} min]", flush=True)
    bad = [r for r in stored if not r['converged']]
    print(f"\n{len(bad)} of {len(stored)} stored nonlinear-solver predictions are not converged at the third basis size:")
    for r in bad:
        keys = {k: r[k] for k in r if k in ('Qp', 'taub', 'damping', 'Nb', 'distribution', 'matching', 'which', 'bucket')}
        print(f"  {r['scan']} {keys}: stored {r['stored'][1]:.5f}, nearest at n3 {r['nearest_n3'][1]:.5f} (dg {100*r['dg_rel']:.1f}%, dRe {r['dRe']:.4f}), held3 {r['held3']}")
    print('done')
