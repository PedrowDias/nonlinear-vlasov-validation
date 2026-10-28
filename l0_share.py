"""Bunch-length paragraph of Sec. V C: the most unstable mode beyond 1.5 ns is "an l = 0 radial mode (99% l = 0 content)".
The eigenvectors of bunch_length.solver_modes at (6,48), tau_b = 1.6, 1.7, 1.8 ns, Q' = +5, both buckets (mode -1 alongside),
and the share of their l = 0 block in two norms: the coefficients, sum_n |c_0n|^2 / sum_ln |c_ln|^2, and the weighted
norm int dJ |R_0|^2/g0 over the sum over all l (the solver's normalization quadrature). Solver only, a few minutes.
Writes data/validation/l0_share.json.

    python3 l0_share.py
"""
import os, json
import numpy as np

import common as c
import bunch_length as bl
from nonlinear import norm_quadrature, norm

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'l0_share.json')
TAUBS = (1.6e-9, 1.7e-9, 1.8e-9)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for taub in TAUBS:
        sigma_z = taub*c.v/4
        for mode, h_b in (('nonlinear', c.h), ('linear', c.h/200.)):
            b = c.gaussian_bucket(bl.nmax_check, sigma_z=sigma_z, h_b=h_b)
            nq = norm_quadrature(bl.lmax, bl.nmax_check, c.R, c.Qs, b['h'], c.eta, b['g0f'], Jz_upper=b['Jz_max'])
            for name, (ev, vec, held) in bl.solver_modes(b, 5.).items():
                share_coef = float(np.sum(np.abs(vec[bl.lmax])**2)/np.sum(np.abs(vec)**2))
                v0 = np.zeros_like(vec); v0[bl.lmax] = vec[bl.lmax]
                share_w = float(norm(v0, nq)**2/norm(vec, nq)**2)
                rows.append(dict(taub_ns=round(taub*1e9, 2), bucket=mode, mode=name, re=ev.real, g=-ev.imag, held=bool(held),
                                 l0_share_coefficients=share_coef, l0_share_weighted=share_w))
                print(f"{taub*1e9:.1f} ns {mode:>9} {name:>10}: dQ {ev.real:+.4f}/{-ev.imag:.5f}{'' if held else ' (not held)'}; "
                      f"l = 0 share: coefficients {100*share_coef:.1f}%, weighted int|R_l|^2/g0 {100*share_w:.1f}%", flush=True)
    json.dump(rows, open(PATH, 'w'), indent=1)
    rad = [r for r in rows if r['mode'] == 'l=0 radial' and r['bucket'] == 'nonlinear']
    print(f"sinusoidal bucket, radial mode, l = 0 share: coefficients {100*min(r['l0_share_coefficients'] for r in rad):.1f}-{100*max(r['l0_share_coefficients'] for r in rad):.1f}%, "
          f"weighted {100*min(r['l0_share_weighted'] for r in rad):.1f}-{100*max(r['l0_share_weighted'] for r in rad):.1f}%")
    print('done')
