"""Bunch-length paragraph of Sec. V C and footnote b of Table I: the long bunches at Q' = +5 (tau_b = 1.6, 1.7, 1.8 ns).
There DELPHI has two modes within 2.4-6.4% of each other in growth rate, mode -1 (Re dQ = -0.98 Qs) and an l = 0 radial mode
(Re dQ = +0.02 Qs); a kick excites both, and no run of reasonable length lets mode -1 dominate. Each of the two modes is
seeded with its own eigenvector (seeded.py) in both buckets, for 10 e-foldings, and fitted by the slope over the
last half (g_fit) and, as a cross-check, by a matrix pencil over the last 90% (g_pencil). The linearized bucket is seeded from
the nonlinear solver in a bucket stretched 200 times (its linear limit) and compared with DELPHI's own spectrum. 12 runs.
The seeded runs of mode -1 in the linearized bucket are the points of Fig. 4(a) at these lengths (bunch_length.py makes the
same runs). Writes data/validation/bunch_length_seeded.json.

    python3 bunch_length_seeded.py      # about 4 h on an Apple M3
"""
import os, json

import common as c
from common import DELPHI
import seeded
import bunch_length as bl

Qp = 5.
taubs = [1.6e-9, 1.7e-9, 1.8e-9]
OUT = os.path.join(c.DATA, 'validation')


def delphi_modes(taub, Qp=Qp, Nb=bl.Nb, lmax=bl.lmax, nmax=bl.nmax, abseps=bl.abseps):
    """DELPHI's eigenvalues dQ/Qs at Q' for a Gaussian bunch of length taub"""
    g, a, b = DELPHI.longdistribution_decomp(taub, typelong='Gaussian')
    omegaksi = Qp*c.omega0/c.eta
    matZ = DELPHI.compute_impedance_matrix(lmax, nmax, 0, 1, omegaksi, c.omega0, c.Qfrac, a, b, taub, g, c.Z, c.freq, abseps=abseps/10)[0]
    matd = DELPHI.compute_damper_matrix(lmax, nmax, 0, 1, omegaksi, c.omega0, c.Qfrac, a, b, taub, g)
    cd, cz = DELPHI.computes_coef(c.f0, 0., b, g[0], 1., taub, 0., 1, Nb, c.gamma, c.Qx)
    return DELPHI.eigenmodesDELPHI(lmax, nmax, matd, matZ, cd, cz, c.omegas)[0]/c.omega0/c.Qs


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'bunch_length_seeded.json')
    rows = []
    for taub in taubs:
        sigma_z = taub*c.v/4
        dm = delphi_modes(taub)
        delphi = {name: sorted((x for x in dm if x.imag < 0 and inside(x)), key=lambda x: x.imag)[:1] for name, inside in bl.WINDOWS.items()}
        print(f"tau_b = {taub*1e9:.1f} ns, DELPHI: " + ", ".join(f"{n} {x[0].real:+.4f}/{-x[0].imag:.5f}" for n, x in delphi.items() if x), flush=True)
        buckets = {'linear': c.gaussian_bucket(bl.nmax_check, sigma_z=sigma_z, h_b=c.h/200.), 'nonlinear': c.gaussian_bucket(bl.nmax_check, sigma_z=sigma_z)}
        for mode, b in buckets.items():
            for name, (ev, vec, held) in bl.solver_modes(b, Qp).items():
                d = delphi[name][0] if (mode == 'linear' and delphi[name]) else None
                r, signal = bl.seeded_run(mode, Qp, taub, b, ev, vec)
                c.save_signal(f"bunch_length_seeded_{taub*1e9:.1f}ns_{mode}_{name.replace(' ', '_').replace('=', '')}", signal)
                re_p, g_p, in_window = seeded.pencil(signal[len(signal)//10:], re_target=ev.real)
                ref = (-d.imag) if d is not None else -ev.imag
                on_mode = abs(r['re'] - ev.real) < 0.02                         # the slope fit sits on the seeded mode (else another took over)
                rows.append(dict(taub=round(taub*1e9, 2), bucket=mode, mode=name, solver=[ev.real, -ev.imag], held=bool(held),
                                 delphi=[d.real, -d.imag] if d is not None else None, g_pencil=g_p, re_pencil=re_p, pencil_in_window=in_window,
                                 g_fit=r['g'], re_fit=r['re'], fit_on_mode=on_mode, turns=r['turns'], seed_sigma=seeded.SEED_SIGMA))
                print(f"  {mode:>9}, {name}: solver {ev.real:+.4f}/{-ev.imag:.5f}{'' if held else ' (not held)'}"
                      + (f", DELPHI {d.real:+.4f}/{-d.imag:.5f}" if d is not None else '')
                      + f" | seeded run: slope fit {r['re']:+.4f}/{r['g']:.5f} ({100*(r['g']/ref - 1):+.2f}%{'' if on_mode else ', another mode'}), "
                      f"pencil {re_p:+.4f}/{g_p:.5f} ({100*(g_p/ref - 1):+.2f}%{'' if in_window else ', no pole near the mode'}), {r['turns']} turns", flush=True)
                with open(path, 'w') as f:
                    json.dump(rows, f, indent=1, default=lambda o: o.item())
    print('done')
