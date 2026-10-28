"""Chromaticity paragraph of Sec. V C: the l = 0 radial mode at N_b = 1e11, Q' = +5, which the sinusoidal bucket changes by
3.5% in the solvers. In the linearized bucket DELPHI's mode -1 grows within 3.3% of it, so a kicked run does not isolate it:
the mode is seeded in both buckets with the nonlinear solver's eigenvector at (6,48) (in the bucket stretched 200 times for
the linearized rf), the eigenvalue nearest the prediction stored by nb_scan.py, with nb_scan.py's machine and draw, for 10
e-foldings (seeded.py). Writes data/validation/nb_l0_seeded.json.

    python3 nb_l0_seeded.py                  # both buckets, about half an hour each on an Apple M3
    python3 nb_l0_seeded.py linear
"""
import os, sys, json, time
import numpy as np
import xwakes as xw

import common as c
import nb_scan as nb
import seeded

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'nb_l0_seeded.json')
Nb = 1e11
LMAX, NMAX = 6, 48
MAX_TURNS = 300_000


def prediction(mode):
    """the most unstable mode stored by nb_scan.py: DELPHI's for the linearized bucket, the nonlinear solver's for the sinusoidal one"""
    row = next(r for r in json.load(open(os.path.join(OUT, 'nb_scan.json')))['solvers'] if r['Nb'] == Nb)
    re, g = row['delphi'] if mode == 'linear' else row['nonlinear_most']
    return complex(re, -g), row


def seed(mode, target):
    """the nonlinear solver's eigenvalue nearest the prediction among the growing ones, at (LMAX, NMAX), its eigenvector, and
    the share of its l = 0 block in the coefficient norm"""
    b = c.gaussian_bucket(NMAX, h_b=seeded.h_of[mode])
    matZ, matd = seeded.seed_matrices(b, LMAX, NMAX, nb.Qp, 3e11)               # line-sum tolerance as nb_scan.predict
    ev, vec = seeded.nearest_mode(b, LMAX, NMAX, matZ, matd, Nb, 0., target.real, -target.imag)
    l_share = float(np.sum(np.abs(vec[LMAX])**2)/np.sum(np.abs(vec)**2))
    return b, ev, vec, l_share


def run(mode):
    target, row = prediction(mode)
    b, ev, vec, l_share = seed(mode, target)
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=nb.num_slices)
    line = c.build({'map': c.one_turn(mode, nb.Qp), 'wake': wake})
    zd = c.boltzmann(nb.num_particles, c.alpha, np.random.default_rng(nb.seed)) if mode == 'nonlinear' else None
    particles = c.matched(line, Nb, nb.num_particles, c.sigma_z, zd, seed=nb.seed)                # nb_scan.run's draw
    x, signal = seeded.run(line, particles, (b, ev, vec, LMAX, NMAX, nb.Qp), -target.imag, nb.kick_sigma, MAX_TURNS)
    re_p, g_p, inside = seeded.pencil(signal[len(signal)//10:], re_target=ev.real)
    c.save_signal(f'nb_l0_seeded_{Nb:.1e}_{mode}', signal)
    g = x['g']
    kicked = next(r for r in json.load(open(os.path.join(OUT, 'nb_scan.json')))['tracking'] if r['Nb'] == Nb and r['bucket'] == mode and r['run'] == 'kicked')
    return dict(Nb=Nb, Qp=nb.Qp, bucket=mode, run='seeded', seed=[ev.real, -ev.imag], seed_l0_share=l_share, gp=-target.imag, rep=target.real,
                g=g, re=x['re'], g_pencil=g_p, re_pencil=re_p, pencil_in_window=bool(inside), turns=x['turns'],
                dev_pct=100*(g/(-target.imag) - 1) if g is not None else None, g_kicked=kicked['g'], dev_kicked_pct=100*(kicked['g']/(-target.imag) - 1),
                delphi_mode_m1=row['delphi_mode_m1'])


if __name__ == '__main__':
    buckets = [a for a in sys.argv[1:] if a in seeded.h_of] or ['nonlinear', 'linear']
    rows = json.load(open(PATH)) if os.path.exists(PATH) else []
    for mode in buckets:
        if any(r['bucket'] == mode for r in rows):
            continue
        t0 = time.time()
        x = run(mode)
        rows.append(x)
        json.dump(rows, open(PATH, 'w'), indent=1, default=lambda o: o.item())
        tracked = 'no growth' if x['g'] is None else f"{x['g']:.6f} ({x['dev_pct']:+.2f}% from the prediction {x['gp']:.6f})"
        print(f"Nb = {Nb:.0e}, {mode}: seeded l = 0 mode {x['seed'][0]:+.4f}/{x['seed'][1]:.6f} (l = 0 share {100*x['seed_l0_share']:.1f}%): "
              f"tracked {tracked}, pencil {x['g_pencil']:.6f}; kicked run {x['dev_kicked_pct']:+.2f}%, {x['turns']} turns "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    by = {r['bucket']: r for r in rows}
    if len(by) == 2:
        print(f"sinusoidal/linearized: seeded {by['nonlinear']['g']/by['linear']['g']:.4f}, kicked {by['nonlinear']['g_kicked']/by['linear']['g_kicked']:.4f}, "
              f"solvers {by['nonlinear']['gp']/by['linear']['gp']:.4f}")
    print('done')
