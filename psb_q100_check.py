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
MODE, QP = 'nonlinear', -1.                            # the run

if __name__ == '__main__':
    t0 = time.time()
    runs = [r for r in json.load(open(os.path.join(c.DATA, 'validation', 'psb_q100.json')))['runs'] if r['Qp'] == QP and r['bucket'] == MODE]
    kicked = next(r for r in runs if r['run'] == 'kicked')
    res = pc.point(MODE, QP, runs)
    m = min(res['distinct'], key=lambda d: abs(d['p'].real - kicked['re']))      # the mode the kicked run grows on (m = 1)
    assert abs(m['p'].real - kicked['re']) < 0.05 and round(m['p'].real) == 1, m['p']
    g, re = psb.run(MODE, QP, -m['p'].imag, m['vec'])
    out = dict(Qp=QP, bucket=MODE, run='seeded (self-consistent eigenvector)', g=g, re=re, gp=-m['p'].imag, rep=m['p'].real,
               dev_pct=100*(g/(-m['p'].imag) - 1), g_kicked=kicked['g'], re_kicked=kicked['re'], dev_kicked_pct=100*(kicked['g']/(-m['p'].imag) - 1),
               other_modes=[[d['p'].real, -d['p'].imag] for d in res['distinct'][:6]])
    json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    print(f"Q' = {QP:+.0f} {MODE}: m = 1 at {m['p'].real:+.4f}/{-m['p'].imag:.5f}; seeded run {g:.5f} (Re {re:+.4f}), {out['dev_pct']:+.2f}%; "
          f"kicked run {kicked['g']:.5f} (Re {kicked['re']:+.4f}), {out['dev_kicked_pct']:+.2f}% [{(time.time() - t0)/60:.0f} min]")
    print('done')
