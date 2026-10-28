"""Mode-coupling paragraph of Sec. V C and footnote a of Table I: just above the threshold the growth rate rises as
(N_b - N_0)^(1/2), so the tracked rates there are very sensitive to the onset N_0. Kicked runs with tmci.py's machine at the
three scan intensities around each threshold (3.70, 3.75, 3.80e11 linearized; 3.50, 3.55, 3.60e11 sinusoidal), five seeds of
the macroparticle draw each (30 runs of up to 24 000 turns, stopped once the centroid reaches 1 cm, fitted over the last 3000
turns, or over the last two thirds if they do not reach it). The onset N_0 of g^2 = k (N_b - N_0) is fitted to the solver
points within 0.3e11 above the threshold (tmci.json's threshold is the intensity where g = 0.02 Q_s, slightly above N_0), and
each tracked rate is converted into the onset shift (g_p^2 - g^2)/k that would explain it. Reads data/tmci.json; writes
data/validation/tmci_seeds.json (finished runs are kept when the script is restarted).

    python3 tmci_seeds.py
"""
import os, json
import numpy as np
import xwakes as xw

import common as c
import tmci

points = {'linear': [3.70, 3.75, 3.80], 'nonlinear': [3.50, 3.55, 3.60]}      # x 1e11; the first of each is just below the threshold
seeds = [1, 2, 3, 4, 5]
num_turns = 24_000              # kicked runs (tmci.py's machine and bunch), stopped once the centroid reaches amp_max
kick_sigma = 0.05
amp_max = 1e-2
fit_turns = 3000
OUT = os.path.join(c.DATA, 'validation')


def run(mode, Nb, seed):
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=tmci.num_slices)
    line = c.build({'map': c.one_turn(mode, tmci.Qp), 'wake': wake})
    zd = c.boltzmann(tmci.num_particles, c.alpha, np.random.default_rng(seed)) if mode == 'nonlinear' else None
    particles = c.matched(line, Nb, tmci.num_particles, c.sigma_z, zd, seed=seed)
    particles.x += kick_sigma*c.sigma_x
    signal = c.track(line, particles, num_turns, amp_max=amp_max)
    i1 = len(signal)
    return c.fit(signal, max(i1//3, i1 - fit_turns) if abs(signal[-1]) > amp_max else i1//3)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    stored = c.load('tmci')
    thresholds = stored['threshold']
    pred = {(r['bucket'], r['Nb']): r for r in stored['rows']}
    # the sqrt law g^2 = k (Nb - N0) from the solver points within 0.3e11 above threshold (least squares)
    onset = {}
    for mode in points:
        pts = np.array([(r['Nb'], r['gp']**2) for r in stored['rows'] if r['bucket'] == mode and r['gp'] and 0 < r['Nb'] - thresholds[mode] <= 0.3])
        k, b0 = np.polyfit(pts[:, 0], pts[:, 1], 1)
        onset[mode] = dict(k=k, N0=-b0/k, Nth_002=thresholds[mode], residual=float(np.abs(pts[:, 1] - (k*pts[:, 0] + b0)).max()/pts[:, 1].max()))
        print(f"{mode}: g^2 = {k:.5f} (Nb - {-b0/k:.4f}e11) from {len(pts)} solver points (max residual {onset[mode]['residual']:.1e}); "
              f"threshold at 0.02 Qs: {thresholds[mode]:.4f}e11", flush=True)

    def equivalent_shift(mode, Nb, g, gp):
        """the threshold shift that would explain g instead of gp, from g^2 = k (Nb - N0): dN = (gp^2 - g^2)/k"""
        return (gp**2 - g**2)/onset[mode]['k'] if gp and g else None

    path = os.path.join(OUT, 'tmci_seeds.json')
    rows = json.load(open(path))['rows'] if os.path.exists(path) else []       # finished runs are kept
    done = {(r['bucket'], r['Nb'], r['seed']) for r in rows}
    for mode, Nbs in points.items():
        for Nb in Nbs:
            p = pred[(mode, Nb)]
            for s in seeds:
                if (mode, Nb, s) in done:
                    continue
                g, re, grew = run(mode, Nb*1e11, s)
                g = g if grew else 0.
                dNth = equivalent_shift(mode, Nb, g, p['gp'])
                rows.append(dict(Nb=Nb, bucket=mode, seed=s, g=g, re=re, gp=p['gp'], rep=p['rep'], Nth=thresholds[mode], N0=onset[mode]['N0'], dNth_equiv=dNth))
                dev = f"{100*(g/p['gp'] - 1):+.1f}%" if p['gp'] and g else 'no growth'
                print(f"Nb = {Nb:.2f}e11, {mode}, seed {s}: growth {g:.5f} ({p['gp']:.5f}, {dev}), Re {re:+.4f} ({p['rep']:+.4f})"
                      + (f"; equivalent threshold shift {dNth:+.4f}e11 ({100*dNth/thresholds[mode]:+.2f}%)" if dNth is not None else ''), flush=True)
                with open(path, 'w') as f:
                    json.dump(dict(thresholds=thresholds, onset=onset, rows=rows), f, indent=1, default=lambda o: o.item())
            gs = np.array([r['g'] for r in rows if r['Nb'] == Nb and r['bucket'] == mode])
            grow = gs[gs > 0]
            print(f"==> Nb = {Nb:.2f}e11, {mode}: {len(grow)} of {len(gs)} seeds grow; over the growing seeds tracked {grow.mean() if len(grow) else 0:.5f} "
                  f"+- {grow.std(ddof=1) if len(grow) > 1 else 0:.5f} against {p['gp']:.5f} predicted", flush=True)
    print('done')
