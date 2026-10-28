"""Damper paragraph of Sec. V C: at Q' = -5 the simulations fall increasingly below both solvers as the damper gain rises.
damper.py applies the wake and the damper once per turn, in this order (map -> wake -> damper); the solvers have neither an
order nor a number of kick points. The scan's runs are repeated with the same macroparticles and seed (damper.machine,
damper.mode_seed, seeded.run) on other machines:
  A    map -> wake -> damper                    damper.py's machine: the control, which reproduces data/damper.json
  B    map -> damper -> wake                    the two kicks swapped
  N2   2 x [map/2 -> wake/2 -> damper/2]        wake and damper spread over 2 points per turn, the map split with them
  N4   4 x [map/4 -> wake/4 -> damper/4]        and over 4
B, N2 and N4 at damping 0, 0.10, 0.125 and 0.15 Q_s in both buckets, each with A at 0.10 Q_s as the control. Each variant also
gets the rigid-bunch calibration of damper_trend.py (no wake, Q' = 0, damping 0.05 and 0.10 Q_s; without wake B is the same
machine as A, so B's calibration is A's), whose ratio to the set rate is -N ln(1 - 2a/N)/(2a) for N kick points,
a = 2 pi Q_s damping, i.e. about 1 + a for N = 1. Writes data/validation/damper_lumped_<variant>_<buckets>.json (one file per
process, so that several can run side by side); `summary` merges them into data/validation/damper_lumped.json.

    python3 damper_lumped.py B               # one variant, both buckets (B, N2, N4 also run A at 0.10 Q_s as the control)
    python3 damper_lumped.py N4 linear       # one variant, one bucket
    python3 damper_lumped.py summary
"""
import os, sys, json, time, glob
import numpy as np
import xtrack as xt
import xwakes as xw
from xfields import TransverseDamper

import common as c
import damper as da
import seeded

OUT = os.path.join(c.DATA, 'validation')
Qp = -5.
DAMPING = (0.0, 0.10, 0.125, 0.15)
CALIB = (0.05, 0.10)
VARIANTS = {'A': (1, 'wake first'), 'B': (1, 'damper first'), 'N2': (2, 'wake first'), 'N4': (4, 'wake first')}
BUCKETS = ('linear', 'nonlinear')


def segment_map(mode, Qp, f):
    """the one-turn map of common.one_turn scaled to a fraction f of the turn (f = 1 gives common.one_turn's element)"""
    common = dict(length=c.circ*f, qx=c.Qx*f, qy=c.Qy*f, betx=c.beta_x, bety=c.R/c.Qy, dqx=Qp*f, dqy=0.)
    if mode == 'linear':
        return xt.LineSegmentMap(longitudinal_mode='linear_fixed_qs', qs=c.Qs*f, bets=c.bets, **common)
    return xt.LineSegmentMap(longitudinal_mode='nonlinear', momentum_compaction_factor=c.alpha_p,
                             voltage_rf=[c.V_rf*f], frequency_rf=[c.h*c.f0], phase_rf=[np.pi], **common)


def variant_line(variant, mode, Qp, damping, wake=True):
    """the machine of `variant`; damping in Qs (the rigid-mode damping rate), per turn d = damping 2 pi Qs"""
    n, order = VARIANTS[variant]
    d = damping*2*np.pi*c.Qs
    zr = (-c.zeta_half_range, c.zeta_half_range)
    el = {}
    for i in range(n):
        f = 1./n
        el[f'map{i}'] = c.one_turn(mode, Qp) if n == 1 else segment_map(mode, Qp, f)
        kicks = []
        if wake:
            w = xw.WakeResonator(kind='dipolar_x', r=c.Rs*f, q=c.Qres, f_r=c.fr)
            w.configure_for_tracking(zeta_range=zr, num_slices=da.num_slices)
            kicks.append((f'wake{i}', w))
        if damping > 0:
            kicks.append((f'damper{i}', TransverseDamper(gain_x=2*d*f, gain_y=0., zeta_range=zr, num_slices=da.num_slices)))
        if order == 'damper first':
            kicks = kicks[::-1]
        el.update(kicks)
    return c.build(el)


def stored_row(mode, damping):
    return next(r for r in c.load('damper') if r['Qp'] == Qp and r['bucket'] == mode and abs(r['damping'] - damping) < 1e-9)


def seeded_run(variant, mode, damping):
    """the scan's point (damper.py: same draw, same seed), tracked on the variant's machine"""
    r = stored_row(mode, damping)
    d = damping*2*np.pi*c.Qs
    line_A, particles = da.machine(mode, Qp, d)
    s = da.mode_seed(mode, Qp, d, r['gp'], r['rep'])
    assert s is not None, f'no growing mode is predicted at damping {damping} ({mode}): nothing to seed'
    line = line_A if variant == 'A' else variant_line(variant, mode, Qp, damping)
    x, signal = seeded.run(line, particles, s, r['gp'], da.kick_sigma, da.max_turns)
    re_p, g_p, inside = seeded.pencil(signal[len(signal)//10:], re_target=s[1].real)
    c.save_signal(f'damper_lumped_{variant}_{Qp}_{damping}_{mode}', signal)
    out = dict(test='seeded', variant=variant, kick_points=VARIANTS[variant][0], order=VARIANTS[variant][1], bucket=mode, Qp=Qp,
               damping=damping, turns=x['turns'], seed=[s[1].real, -s[1].imag], g=x['g'], re=x['re'],
               g_pencil=g_p, re_pencil=re_p, pencil_in_window=bool(inside), gp=r['gp'], rep=r['rep'], g_stored=r['g'],
               dev_pct=100*(x['g']/r['gp'] - 1) if x['g'] is not None else None)
    if variant == 'A':                                        # the control: damper.py's own run
        out['identical_to_stored_g'] = bool(x['g'] == r['g'])
    return out


def calibration(variant, mode, damping, particles=100_000, turns=20_000):
    """damper_trend.calibration on the variant's machine without wake: decay rate of a kicked rigid bunch, Q' = 0, in Qs"""
    d = damping*2*np.pi*c.Qs
    zr = (-c.zeta_half_range, c.zeta_half_range)
    line_A = c.build({'map': c.one_turn(mode, 0.), 'damper': TransverseDamper(gain_x=2*d, gain_y=0., zeta_range=zr, num_slices=da.num_slices)})
    zd = c.boltzmann(particles, c.alpha, np.random.default_rng(da.seed)) if mode == 'nonlinear' else None
    p = c.matched(line_A, da.Nb, particles, c.sigma_z, zd)          # as damper_trend.calibration: A reproduces damper_trend.json
    p.x += 0.1*c.sigma_x
    line = line_A if variant == 'A' else variant_line(variant, mode, 0., damping, wake=False)
    signal = c.track(line, p, turns)
    n = int(min(turns, 3/(2*np.pi*c.Qs*damping)))
    rate = -np.polyfit(np.arange(n), np.log(np.abs(signal[:n])), 1)[0]/(2*np.pi*c.Qs)
    a = 2*np.pi*c.Qs*damping
    N = VARIANTS[variant][0]
    return dict(test='calibration', variant=variant, kick_points=N, order=VARIANTS[variant][1], bucket=mode, Qp=0., damping=damping,
                rate=rate, ratio=rate/damping, ratio_expected=-N*np.log(1 - 2*a/N)/(2*a))


def summary():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, 'damper_lumped_*.json'))):
        if os.path.basename(f) != 'damper_lumped.json':
            rows += json.load(open(f))
    seen, merged = set(), []
    for r in rows:                                            # the A controls appear once per process
        k = (r['test'], r['variant'], r['bucket'], r['damping'])
        if k not in seen:
            seen.add(k); merged.append(r)
    json.dump(merged, open(os.path.join(OUT, 'damper_lumped.json'), 'w'), indent=1, default=lambda o: o.item())
    for mode in BUCKETS:
        print(f"{mode} bucket, Q' = {Qp:+.0f}: tracked/solver - 1 (%)")
        print('  damping  stored    ' + '  '.join(f'{v:>7}' for v in VARIANTS))
        for d in DAMPING:
            st = stored_row(mode, d)
            vals = []
            for v in VARIANTS:
                x = next((r for r in merged if r['test'] == 'seeded' and r['variant'] == v and r['bucket'] == mode and r['damping'] == d), None)
                vals.append('      -' if x is None or x['dev_pct'] is None else f"{x['dev_pct']:+7.2f}")
            print(f"  {d:6.3f}  {100*(st['g']/st['gp'] - 1):+7.2f}   " + '  '.join(vals))
        for d in CALIB:
            vals = []
            for v in VARIANTS:
                x = next((r for r in merged if r['test'] == 'calibration' and r['variant'] == v and r['bucket'] == mode and r['damping'] == d), None)
                vals.append('      -' if x is None else f"{x['ratio']:.5f}/{x['ratio_expected']:.5f}")
            print(f"  calibration {d:.2f} Qs, measured/expected ratio: " + '  '.join(vals))
    ctl = [r for r in merged if r['variant'] == 'A' and r['test'] == 'seeded']
    print('A controls identical to the scan (data/damper.json):', [(r['bucket'], r['damping'], r['identical_to_stored_g']) for r in ctl])


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['summary']:
        summary(); sys.exit()
    if not args or args[0] not in VARIANTS:
        sys.exit(f"usage: python3 damper_lumped.py {'|'.join(VARIANTS)} [linear|nonlinear], or python3 damper_lumped.py summary")
    variant = args[0]
    buckets = [a for a in args[1:] if a in BUCKETS] or list(BUCKETS)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"damper_lumped_{variant}_{'_'.join(buckets)}.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r['test'], r['variant'], r['bucket'], r['damping']) for r in rows}
    jobs = [('seeded', 'A', m, 0.10) for m in buckets] if variant != 'A' else []
    jobs += [('seeded', variant, m, d) for m in buckets for d in DAMPING]
    jobs += [('calibration', 'A' if variant == 'B' else variant, m, d) for m in buckets for d in CALIB]   # B without wake is A
    for test, v, mode, d in jobs:
        if (test, v, mode, d) in done:
            continue
        t0 = time.time()
        x = seeded_run(v, mode, d) if test == 'seeded' else calibration(v, mode, d)
        rows.append(x)
        json.dump(rows, open(path, 'w'), indent=1, default=lambda o: o.item())
        if test == 'seeded':
            extra = f", identical to the scan: {x['identical_to_stored_g']}" if v == 'A' else ''
            print(f"{v:>2} {mode:>9} damping {d:.3f} Qs: tracked {x['g']:.5f} (pencil {x['g_pencil']:.5f}) vs predicted {x['gp']:.5f}: "
                  f"{x['dev_pct']:+.2f}% (stored run {100*(x['g_stored']/x['gp'] - 1):+.2f}%), {x['turns']} turns{extra} [{(time.time()-t0)/60:.1f} min]", flush=True)
        else:
            print(f"{v:>2} {mode:>9} calibration {d:.2f} Qs: rate {x['rate']:.6f}, ratio {x['ratio']:.6f} (once-per-point expectation "
                  f"{x['ratio_expected']:.6f}) [{(time.time()-t0)/60:.1f} min]", flush=True)
    print('done')
