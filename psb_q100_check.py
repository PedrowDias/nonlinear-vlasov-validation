"""PSB paragraph of Sec. V E, Q_r = 100: the one run of the 50 that differs from the evaluation at its own frequency by more
than 1.1%, Q' = -1 in the sinusoidal bucket, a kicked run that grows on m = 1. This seeds m = 1 there with its self-consistent
eigenvector (psb_q100_candidates.point, then psb.run with Q_r = 100 and 147 turns of wake memory), which tells a property of
the kicked start from a property of the mode. Writes data/validation/psb_q100_check.json. About 20 min on an Apple M3.

    python3 psb_q100_check.py
"""
import os, json, time

import common as c
import psb_q100_candidates as pc
import psb

PATH = os.path.join(c.DATA, 'validation', 'psb_q100_check.json')
MODE, QP, RE_M1 = 'nonlinear', -1., 1.149              # the run, and the Re dQ/Qs of the mode m = 1 it grows on

if __name__ == '__main__':
    t0 = time.time()
    runs = [r for r in json.load(open(os.path.join(c.DATA, 'validation', 'psb_q100.json')))['runs'] if r['Qp'] == QP and r['bucket'] == MODE]
    res = pc.point(MODE, QP, runs)
    m = min(res['distinct'], key=lambda d: abs(d['p'].real - RE_M1))
    assert abs(m['p'].real - RE_M1) < 0.01, m['p']
    g, re = psb.run(MODE, QP, -m['p'].imag, m['vec'])
    kicked = runs[0]
    out = dict(Qp=QP, bucket=MODE, run='seeded (self-consistent eigenvector)', g=g, re=re, gp=-m['p'].imag, rep=m['p'].real,
               dev_pct=100*(g/(-m['p'].imag) - 1), g_kicked=kicked['g'], re_kicked=kicked['re'], dev_kicked_pct=100*(kicked['g']/(-m['p'].imag) - 1),
               other_modes=[[d['p'].real, -d['p'].imag] for d in res['distinct'][:6]])
    json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    print(f"Q' = {QP:+.0f} {MODE}: m = 1 at {m['p'].real:+.4f}/{-m['p'].imag:.5f}; seeded run {g:.5f} (Re {re:+.4f}), {out['dev_pct']:+.2f}%; "
          f"kicked run {kicked['g']:.5f} (Re {kicked['re']:+.4f}), {out['dev_kicked_pct']:+.2f}% [{(time.time() - t0)/60:.0f} min]")
    print('done')
