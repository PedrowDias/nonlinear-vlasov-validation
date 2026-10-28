"""PSB paragraph of Sec. V E, Q_r = 100: the PSB scan of psb.py with the resonator's quality factor of DELPHI's own PSB test
case, Q_r = 100 instead of 10. Everything else as psb.py (parabolic bunch of 503 ns, N_b = 4e12, Q' = -8 to +4, both buckets,
(13,32) held against (13,24), the impedance at the betatron lines and at each mode's own frequency, every mode within 8% of the
fastest seeded with its eigenvector, otherwise kicked, 25 e-foldings), except the wake memory: 147 turns instead of 20, for
the same 1e-3 cut-off of a wake that decays ten times more slowly. Writes data/validation/psb_q100.json (finished runs are
kept when the script is restarted).

    python3 psb_q100.py
"""
import os, json, time
import numpy as np

import common as c
import psb
from nonlinear import resonator_impedance

QRES = 100.
OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'psb_q100.json')

psb.Qres = QRES                                              # psb.sampled_at, psb.eigenmodes (through psb.Z) and psb.run read the module's Qres
psb.Z = resonator_impedance(psb.Rt, psb.fr, psb.Qres, psb.freq)
# The wake of a resonator decays as exp(-pi f_r t/Q_r): with Q_r = 10 it is below 2e-5 after psb.py's 20 turns of wake memory,
# with Q_r = 100 it is still 34% there, so the memory is set for the same 1e-3 cut-off: n > Q_r ln(1000)/(pi f_r T_0) = 126.8 turns,
# i.e. 127, plus the 20 of psb.py.
psb.num_turns_wake = int(np.ceil(QRES*np.log(1e3)/(np.pi*psb.fr/psb.f0))) + 20          # 147


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    out = json.load(open(PATH)) if os.path.exists(PATH) else dict(Qres=QRES, lmax=psb.lmax, nmax=psb.nmax, nmax_check=psb.nmax_check, num_turns_wake=psb.num_turns_wake, runs=[])
    done = {(r['Qp'], r['bucket'], round(r['rep_line'], 4)) for r in out['runs']}
    for Qp in psb.Qpscan:
        for mode in psb.buckets:
            t0 = time.time()
            group = psb.predict(mode, Qp)
            seeded = len(group) > 1
            for p, pc, vec in group:
                if (Qp, mode, round(p.real, 4)) in done:
                    continue
                g, re = psb.run(mode, Qp, -pc.imag, vec if seeded else None)
                row = dict(Qp=float(Qp), bucket=mode, run='seeded' if seeded else 'kicked', g=g, re=re,
                           gp=-pc.imag, rep=pc.real, gp_line=-p.imag, rep_line=p.real,
                           dev_mode_pct=100*(g/(-pc.imag) - 1), dev_line_pct=100*(g/(-p.imag) - 1))
                out['runs'].append(row)
                print(f"Q' = {Qp:+.0f}, {mode:>9}, {'seeded' if seeded else 'kicked'}: tracked {g:.5f} (Re {re:+.4f}) | solver at the mode's frequency {-pc.imag:.5f} "
                      f"({row['dev_mode_pct']:+.2f}%), at the betatron lines {-p.imag:.5f} ({row['dev_line_pct']:+.2f}%) [{(time.time()-t0)/60:.1f} min]", flush=True)
                json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    runs = out['runs']
    for mode in psb.buckets:
        dl = [r['dev_line_pct'] for r in runs if r['bucket'] == mode]; dm = [r['dev_mode_pct'] for r in runs if r['bucket'] == mode]
        below = [100*(1 - 1/(1 + x/100)) for x in dl]                        # solver at the lines, in % below the tracked rate
        print(f"{mode:>9}: tracking against the solver at the betatron lines {min(dl):+.1f} to {max(dl):+.1f}%, i.e. the lines are "
              f"{min(below):.1f} to {max(below):.1f}% below the simulations; at the modes' own frequencies {min(dm):+.1f} to {max(dm):+.1f}%")
    print('done')
