"""Table I (footnote c) and Sec. V F: DELPHI against the linear limit of the nonlinear solver for the eight coupled-bunch
modes. DELPHI's l = 0 mode of each n_x (the one the resonator shifts, selected as in coupled_bunch.predict), with the setup of
coupled_bunch.py, at (6,48) [and (6,32) with `both`], against the linearized-bucket predictions of data/coupled_bunch.json (the
nonlinear solver in a bucket stretched 200 times). Solver only, about an hour on an Apple M3 (the line sum runs over every
line up to 5 GHz). Writes data/validation/cb_delphi.json.

    python3 cb_delphi.py          # (6,48)
    python3 cb_delphi.py both     # (6,48) and (6,32)
"""
import os, sys, json, time
import numpy as np

import common as c
from common import DELPHI
import coupled_bunch as cb

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'cb_delphi.json')


def delphi_mode(nx, nmax):
    """DELPHI's eigenvalue dQ/Qs of the l = 0 mode the resonator shifts in coupled-bunch mode nx (same selection as
    coupled_bunch.predict: among the eigenvalues with |Re dQ/Qs| < 0.5, the one with the largest |Im|)"""
    g, a, b = DELPHI.longdistribution_decomp(c.taub, typelong='Gaussian')
    omegaksi = cb.Qp*c.omega0/c.eta
    matZ = DELPHI.compute_impedance_matrix(cb.lmax, nmax, nx, cb.M, omegaksi, c.omega0, c.Qfrac, a, b, c.taub, g, cb.Z, cb.freq, abseps=cb.abseps/10)[0]
    matd = DELPHI.compute_damper_matrix(cb.lmax, nmax, nx, cb.M, omegaksi, c.omega0, c.Qfrac, a, b, c.taub, g)
    cd, cz = DELPHI.computes_coef(c.f0, 0., b, g[0], 1., c.taub, 0., cb.M, cb.Nb, c.gamma, c.Qx)
    e = DELPHI.eigenmodesDELPHI(cb.lmax, nmax, matd, matZ, cd, cz, c.omegas)[0]/c.omega0/c.Qs
    near_zero = np.flatnonzero(np.abs(e.real) < 0.5)
    k = near_zero[np.argmax(np.abs(e[near_zero].imag))]
    return e[k]


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    stored = {r['nx']: r for r in c.load('coupled_bunch') if r['bucket'] == 'linear'}
    sizes = [48, 32] if 'both' in sys.argv[1:] else [48]
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    for nmax in sizes:
        key = f'(6,{nmax})'
        rows = out.get(key, [])
        done = {r['nx'] for r in rows}
        for nx in cb.nxscan:
            if int(nx) in done:
                continue
            t0 = time.time()
            x = delphi_mode(int(nx), nmax)
            s = stored[int(nx)]
            rows.append(dict(nx=int(nx), re=x.real, g=-x.imag, re_stored=s['rep'], g_stored=s['gp'],
                             dre=x.real - s['rep'], dg=-x.imag - s['gp'], dg_rel=(-x.imag - s['gp'])/s['gp']))
            print(f"DELPHI {key} nx = {nx}: Re {x.real:+.7f}, growth {-x.imag:+.7f} | stored (stretched nonlinear solver) {s['rep']:+.7f}, {s['gp']:+.7f} "
                  f"| dRe {x.real - s['rep']:+.1e}, dg {-x.imag - s['gp']:+.1e} ({100*(-x.imag - s['gp'])/s['gp']:+.4f}%) [{(time.time()-t0)/60:.1f} min]", flush=True)
            out[key] = rows
            json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
        worst_re = max(abs(r['dre']) for r in rows); worst_g = max(abs(r['dg']) for r in rows); worst_rel = max(abs(r['dg_rel']) for r in rows)
        out[key + ' summary'] = dict(max_abs_dre=worst_re, max_abs_dg=worst_g, max_rel_dg=worst_rel)
        print(f"{key}: over the eight modes, |dRe| <= {worst_re:.1e} Qs, |dg| <= {worst_g:.1e} Qs ({100*worst_rel:.4f}% relative)", flush=True)
        json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    print('done')
