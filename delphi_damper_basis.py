"""Damper paragraph of Sec. V C: "the predictions change by less than 1% between n_max = 32, 48 and 64 (DELPHI's not at
all)". convergence_band.py does the nonlinear solver; this does DELPHI, which damper.py runs at (6,32): its most unstable
eigenvalue at (6,32), (6,48) and (6,64) for Q' = -5 and damping 0, 0.10, 0.125 and 0.15 Q_s (common.delphi_most, as in
damper.py). Solver only; (6,64) needs about twice the memory of (6,48).
Writes data/validation/delphi_damper_basis.json.

    python3 delphi_damper_basis.py
"""
import os, json, time
import numpy as np

import common as c
import damper as da

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'delphi_damper_basis.json')
Qp = -5.
damping = np.array([0., 0.10, 0.125, 0.15])
SIZES = (32, 48, 64)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    dmax = damping*2*np.pi*c.Qs
    stored = json.load(open(PATH))['rows'] if os.path.exists(PATH) else []
    eig = {}
    for nmax in SIZES:
        key = f'eig{nmax}'
        if len(stored) == len(damping) and all(key in r for r in stored):
            eig[nmax] = np.array([complex(r[key][0], -r[key][1]) for r in stored])
            print(f"DELPHI (6,{nmax}): stored", flush=True)
            continue
        t0 = time.time()
        eig[nmax] = c.delphi_most([Qp], dmax, da.Nb, c.taub, da.lmax, nmax)[0]            # dQ/Qs per damper rate
        print(f"DELPHI (6,{nmax}): " + ', '.join(f"d = {d:.3f}: {x.real:+.4f}/{-x.imag:.5f}" for d, x in zip(damping, eig[nmax])) + f" [{(time.time()-t0)/60:.1f} min]", flush=True)
    rows = []
    for i, d in enumerate(damping):
        row = dict(Qp=Qp, damping=float(d))
        for nmax in SIZES:
            row[f'eig{nmax}'] = [eig[nmax][i].real, -eig[nmax][i].imag]
        a = eig[32][i]
        for nmax in SIZES[1:]:
            b = eig[nmax][i]
            sfx = '' if nmax == 48 else f'_{nmax}'
            row['dg_rel_pct' + sfx] = 100*(b.imag/a.imag - 1)
            row['dre' + sfx] = b.real - a.real
        rows.append(row)
        print(f"d = {d:.3f} Qs: growth (6,32) {-a.imag:.6f}, (6,48) {-eig[48][i].imag:.6f}, (6,64) {-eig[64][i].imag:.6f}; change from (6,32): "
              f"{row['dg_rel_pct']:+.2e}%, {row['dg_rel_pct_64']:+.2e}%", flush=True)
    worst = max(max(abs(r['dg_rel_pct']), abs(r['dg_rel_pct_64'])) for r in rows)
    worst_re = max(max(abs(r['dre']), abs(r['dre_64'])) for r in rows)
    json.dump(dict(what="DELPHI's most unstable mode at Q' = -5 with the damper, (6,32) vs (6,48) and (6,64)", lmax=da.lmax, Nb=da.Nb, rows=rows,
                   max_change_pct=worst, max_change_re=worst_re), open(PATH, 'w'), indent=1)
    print(f"largest change from (6,32): growth {worst:.2e}%, Re {worst_re:.1e} Qs")
    print('done')
