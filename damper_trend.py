"""Damper paragraph of Sec. V C: is the growing deviation of the sinusoidal bucket at Q' = -5 a matter of resolution, and
does the simulation's damper damp at the set rate? Kicked runs with the machine and draw of damper.py:
  (1) Q' = -5, damping 0.15 Q_s, at the scan's resolution (1e5 macroparticles, 300 slices) and with 600 slices;
  (2) Q' = -5, damping 0.10, 0.125 and 0.15 Q_s, and Q' = +5, 0.15 Q_s, with four times the particles and twice the slices;
  (3) calibration: no wake, Q' = 0, damping 0.05 and 0.10 Q_s, both buckets: the decay rate of a kicked rigid bunch against
      the set rate d (a kick once per turn damps at -ln(1 - 2a)/(2a) times d, a = 2 pi Q_s d; see damper_lumped.py).
Writes data/validation/damper_trend.json (finished rows are kept when the script is restarted).

    python3 damper_trend.py            # everything (about half a day on an Apple M3)
    python3 damper_trend.py calib      # the calibration only (minutes)
"""
import os, sys, json, time
import numpy as np
import xwakes as xw
from xfields import TransverseDamper

import common as c
import damper as da

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'damper_trend.json')

# the kicked runs: damper.py's machine and draw (Nb, seed, kick amplitude), tracked for 12 e-foldings of the predicted growth
# (at most 1e5 turns) or until the centroid reaches 1 mm, and fitted over the last quarter but at least the last 3000 turns
# (over the last two thirds if the run does not reach 1 mm)
efoldings, max_turns, amp_max, fit_turns = 12, 100_000, 1e-3, 3000

TESTS = [  # (label, Qp, damping [Qs], particles, slices)
    ('repro',      -5., 0.150, 100_000, 300),
    ('slices',     -5., 0.150, 100_000, 600),
    ('resolution', -5., 0.150, 400_000, 600),
    ('resolution', -5., 0.125, 400_000, 600),
    ('resolution', -5., 0.100, 400_000, 600),
    ('resolution', +5., 0.150, 400_000, 600),
]


def predicted(Qp, damping):
    rows = c.load('damper')
    q = [r for r in rows if r['bucket'] == 'nonlinear' and r['Qp'] == Qp and abs(r['damping'] - damping) < 1e-9][0]
    return q['gp'], q['rep']


def run(Qp, damping, particles, slices, growth_predicted):
    """a kicked run on damper.py's machine and draw, with the number of macroparticles and of slices as arguments"""
    d = damping*2*np.pi*c.Qs
    zr = (-c.zeta_half_range, c.zeta_half_range)
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=zr, num_slices=slices)
    damper = TransverseDamper(gain_x=2*d, gain_y=0., zeta_range=zr, num_slices=slices)
    line = c.build({'map': c.one_turn('nonlinear', Qp), 'wake': wake, 'damper': damper})
    zd = c.boltzmann(particles, c.alpha, np.random.default_rng(da.seed))
    p = c.matched(line, da.Nb, particles, c.sigma_z, zd)
    p.x += da.kick_sigma*c.sigma_x
    signal = c.track(line, p, c.turns(efoldings, growth_predicted, max_turns), amp_max=amp_max)
    i1 = len(signal)
    i0 = max(i1//3, i1 - max(fit_turns, i1//4)) if abs(signal[-1]) > amp_max else i1//3
    g, re, grew = c.fit(signal, i0)
    c.save_signal(f'damper_trend_Qp{Qp:+.0f}_d{damping:.3f}_{particles}_{slices}', signal)
    # a second fit over the last half, to show the sensitivity to the window
    g2 = np.polyfit(np.arange(i1//2, i1), np.log(np.abs(signal[i1//2:])), 1)[0]/(2*np.pi*c.Qs)
    return g, re, grew, g2, i1


def calibration(mode, damping, particles=100_000, slices=300, turns=20_000):
    """decay rate of a kicked rigid bunch with the damper alone (no wake), Q' = 0, in units of Qs; expected = damping"""
    d = damping*2*np.pi*c.Qs
    zr = (-c.zeta_half_range, c.zeta_half_range)
    damper = TransverseDamper(gain_x=2*d, gain_y=0., zeta_range=zr, num_slices=slices)
    line = c.build({'map': c.one_turn(mode, 0.), 'damper': damper})
    zd = c.boltzmann(particles, c.alpha, np.random.default_rng(da.seed)) if mode == 'nonlinear' else None
    p = c.matched(line, da.Nb, particles, c.sigma_z, zd)
    p.x += 0.1*c.sigma_x
    signal = c.track(line, p, turns)
    n = int(min(turns, 3/(2*np.pi*c.Qs*damping)))               # fit over ~3 e-foldings of decay (before the noise floor)
    rate = -np.polyfit(np.arange(n), np.log(np.abs(signal[:n])), 1)[0]/(2*np.pi*c.Qs)
    return rate


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open(PATH)) if os.path.exists(PATH) else []
    done = {(r['test'], r['Qp'], r['damping'], r['particles'], r['slices']) for r in rows if r['test'] != 'calibration'}
    if 'calib' in sys.argv[1:] or not [r for r in rows if r['test'] == 'calibration']:
        for mode in ('linear', 'nonlinear'):
            for damping in (0.05, 0.10):
                t0 = time.time()
                rate = calibration(mode, damping)
                rows = [r for r in rows if not (r['test'] == 'calibration' and r['bucket'] == mode and r['damping'] == damping)]
                rows.append(dict(test='calibration', bucket=mode, Qp=0., damping=damping, rate=rate, ratio=rate/damping))
                print(f"calibration {mode:>9} d = {damping:.2f} Qs: rigid bunch decays at {rate:.5f} Qs (ratio {rate/damping:.4f}) [{(time.time()-t0)/60:.1f} min]", flush=True)
                json.dump(rows, open(PATH, 'w'), indent=1, default=lambda o: o.item())
        if 'calib' in sys.argv[1:]:
            sys.exit()
    for test, Qp, damping, particles, slices in TESTS:
        if (test, Qp, damping, particles, slices) in done:
            continue
        gp, rep = predicted(Qp, damping)
        t0 = time.time()
        g, re, grew, g2, n = run(Qp, damping, particles, slices, gp)
        rows.append(dict(test=test, bucket='nonlinear', Qp=Qp, damping=damping, particles=particles, slices=slices, turns=n,
                         g=g if grew else None, g_last_half=g2, re=re, gp=gp, rep=rep, dev=100*(g/gp - 1) if grew else None))
        print(f"{test:>10} Q' = {Qp:+.0f}, d = {damping:.3f} Qs, {particles} particles, {slices} slices: growth {g:.5f} ({100*(g/gp-1):+.2f}% of the solver's {gp:.5f}), "
              f"last-half fit {g2:.5f}, Re {re:+.4f} ({rep:+.4f}), {n} turns [{(time.time()-t0)/60:.0f} min]", flush=True)
        json.dump(rows, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    print('done')
