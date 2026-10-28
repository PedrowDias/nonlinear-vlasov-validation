"""PSB paragraph of Sec. V E, Q_r = 100: the evaluation at the modes' own frequencies against the simulations. psb.predict
iterates the impedance to the mode's own frequency only for the modes within 12% of the fastest at the betatron lines; with
Q_r = 100 the lines underestimate different modes by different amounts, so the mode that is fastest at its own frequency can be
left out. Here every held mode with at least 40% of the fastest growth at the lines is iterated to self-consistency, following
its eigenvector, and every run of psb_q100.json is also iterated from its own tracked tune shift. Then:
  1. the modes at their own frequencies per (Q', bucket), and those within 8% of the fastest (psb.py's seeding rule);
  2. each of the 42 runs of psb_q100.json matched to the self-consistent mode nearest its tracked Re dQ;
  3. every mode of a group of two or more that psb_q100.json has not run seeded is tracked, seeded with its self-consistent
     eigenvector (psb.run with Q_r = 100 and 147 turns of wake memory, set by importing psb_q100).
Writes data/validation/psb_q100_candidates_<bucket>.json; `summary` merges them into psb_q100_candidates.json.

    python3 psb_q100_candidates.py linear
    python3 psb_q100_candidates.py nonlinear
    python3 psb_q100_candidates.py summary
"""
import os, sys, json, time
import numpy as np

import common as c
import psb_q100  # noqa: F401                      # sets psb.Qres = 100, psb.Z and psb.num_turns_wake = 147
import psb
import nonlinear

OUT = os.path.join(c.DATA, 'validation')
FLOOR = 0.4                                         # candidates: held line modes with at least this fraction of the fastest growth
TOL, MAXIT = 1e-6, 30
assert psb.Qres == 100. and psb.num_turns_wake == 147


def solve(b, Qp, n, sampling, vectors=False):
    nonlinear.Zy_interp = sampling
    try:
        return psb.eigenmodes(b, Qp, n, vectors=vectors)
    finally:
        nonlinear.Zy_interp = psb.line_sampling


def unit_columns(vec):
    V = vec.reshape(-1, vec.shape[-1])
    return V/np.linalg.norm(V, axis=0)


def follow(b, Qp, p0, v0=None):
    """fixed point p = eigenvalue of the matrix with Z sampled at p, following the eigenvector from (p0, v0); v0 None: start
    from the eigenvalue nearest p0 of the first matrix (a start from a tracked tune shift)"""
    p, v = p0, (None if v0 is None else v0/np.linalg.norm(v0))
    hist, steps = [[p.real, -p.imag]], []
    for _ in range(MAXIT):
        e, vec = solve(b, Qp, psb.nmax_check, psb.sampled_at(p), vectors=True)
        V = unit_columns(vec)
        k = int(np.argmin(np.abs(e - p))) if v is None else int(np.argmax(np.abs(v.conj() @ V)))
        overlap = 1. if v is None else float(np.abs(v.conj() @ V[:, k]))
        p_new, v, vec_k = e[k], V[:, k], vec[:, :, k]
        steps.append(abs(p_new - p)); hist.append([p_new.real, -p_new.imag])
        p = p_new
        if steps[-1] < TOL:
            break
    e_chk, _ = solve(b, Qp, psb.nmax, psb.sampled_at(p))
    return dict(p=p, vec=vec_k, converged=bool(steps[-1] < TOL), steps=len(steps), last_step=float(steps[-1]),
                last_overlap=overlap, held=bool(c.held(p, e_chk)), history=hist)


def point(mode, Qp, runs):
    b = psb.buckets[mode]
    e, vec = solve(b, Qp, psb.nmax_check, psb.line_sampling, vectors=True)
    e_chk, _ = solve(b, Qp, psb.nmax, psb.line_sampling)
    held = [k for k in np.argsort(e.imag) if e[k].imag < 0 and c.held(e[k], e_chk)]
    cands = [k for k in held if e[k].imag <= FLOOR*e[held[0]].imag]
    modes = []
    for k in cands:
        f = follow(b, Qp, e[k], unit_columns(vec)[:, k])
        modes.append(dict(start='line', start_dQ=[e[k].real, -e[k].imag], **f))
    for r in runs:                                   # from each tracked tune shift as well
        f = follow(b, Qp, r['re'] - 1j*r['g'])
        modes.append(dict(start='tracked', start_dQ=[r['re'], r['g']], **f))
    # distinct fixed points (several starts reach the same one)
    distinct = []
    for m in sorted((m for m in modes if m['converged'] and m['held'] and m['p'].imag < 0), key=lambda m: m['p'].imag):
        if all(abs(m['p'] - d['p']) > 1e-4 for d in distinct):
            distinct.append(m)
    fastest = distinct[0]['p'].imag if distinct else 0.
    group = [d for d in distinct if d['p'].imag <= (1 - psb.within)*fastest]
    return dict(line_held=[[e[k].real, -e[k].imag] for k in held], modes=modes, distinct=distinct, group=group)


def jsonable(m):
    return {k: ([m['p'].real, -m['p'].imag] if k == 'p' else v) for k, v in m.items() if k != 'vec'}


def run_bucket(mode):
    path = os.path.join(OUT, f'psb_q100_candidates_{mode}.json')
    stored = json.load(open(os.path.join(OUT, 'psb_q100.json')))['runs']
    out = json.load(open(path)) if os.path.exists(path) else dict(Qres=psb.Qres, num_turns_wake=psb.num_turns_wake, floor=FLOOR, within=psb.within, points=[], runs=[])
    done = {p['Qp'] for p in out['points']}
    for Qp in psb.Qpscan:
        if float(Qp) in done:
            continue
        t0 = time.time()
        runs = [r for r in stored if r['Qp'] == Qp and r['bucket'] == mode]
        res = point(mode, Qp, runs)
        matched = []
        for r in runs:
            m = min(res['distinct'], key=lambda d: abs(d['p'].real - r['re']))
            matched.append(dict(run=r['run'], g=r['g'], re=r['re'], stored_gp=r['gp'], stored_rep=r['rep'], stored_dev_pct=r['dev_mode_pct'],
                                gp=-m['p'].imag, rep=m['p'].real, dre=r['re'] - m['p'].real, dev_pct=100*(r['g']/(-m['p'].imag) - 1),
                                gp_line_stored=r['gp_line']))
        seeded_rep = [r['rep'] for r in runs if r['run'] == 'seeded']
        new = []
        if len(res['group']) > 1:                    # psb.py's rule: every mode within 8% of the fastest is seeded
            for m in res['group']:
                if any(abs(x - m['p'].real) < 0.02 for x in seeded_rep):
                    continue
                t1 = time.time()
                g, re = psb.run(mode, Qp, -m['p'].imag, m['vec'])
                new.append(dict(Qp=float(Qp), bucket=mode, run='seeded (self-consistent eigenvector)', g=g, re=re, gp=-m['p'].imag, rep=m['p'].real,
                                dev_pct=100*(g/(-m['p'].imag) - 1)))
                print(f"   Q' = {Qp:+.0f} {mode}: seeded mode at {m['p'].real:+.4f}/{-m['p'].imag:.5f}: tracked {g:.5f} (Re {re:+.4f}), "
                      f"{new[-1]['dev_pct']:+.2f}% [{(time.time() - t1)/60:.1f} min]", flush=True)
        out['points'].append(dict(Qp=float(Qp), bucket=mode, line_held=res['line_held'][:30], modes=[jsonable(m) for m in res['modes']],
                                  distinct=[jsonable(m) for m in res['distinct']], group=[jsonable(m) for m in res['group']],
                                  matched=matched))
        out['runs'] += new
        json.dump(out, open(path, 'w'), indent=1, default=lambda o: o.item())
        fmt = lambda ms: ', '.join(f"{m['p'].real:+.3f}/{-m['p'].imag:.4f}" for m in ms)
        print(f"Q' = {Qp:+.0f} {mode}: {len(res['modes'])} iterations, own-frequency modes {fmt(res['distinct'][:6])}; group {fmt(res['group'])}; "
              + '; '.join(f"run {x['run']} {x['g']:.5f} (Re {x['re']:+.3f}) vs {x['gp']:.5f} (Re {x['rep']:+.3f}): {x['dev_pct']:+.2f}% "
                          f"[stored {x['stored_dev_pct']:+.2f}%]" for x in matched) + f" [{(time.time()-t0)/60:.1f} min]", flush=True)


def summary():
    pts, runs = [], []
    for mode in psb.buckets:
        f = os.path.join(OUT, f'psb_q100_candidates_{mode}.json')
        if os.path.exists(f):
            d = json.load(open(f)); pts += d['points']; runs += d['runs']
    json.dump(dict(Qres=psb.Qres, num_turns_wake=psb.num_turns_wake, floor=FLOOR, within=psb.within, points=pts, runs=runs),
              open(os.path.join(OUT, 'psb_q100_candidates.json'), 'w'), indent=1)
    m = [x for p in pts for x in p['matched']]
    print(f"{len(m)} stored runs matched; deviation from the self-consistent mode nearest in Re: {min(x['dev_pct'] for x in m):+.2f} to {max(x['dev_pct'] for x in m):+.2f}%; "
          f"largest |dRe| {max(abs(x['dre']) for x in m):.3f} Qs")
    for x in sorted(m, key=lambda x: -abs(x['dev_pct']))[:8]:
        p = next(p for p in pts if x in p['matched'])
        print(f"  Q' = {p['Qp']:+.0f} {p['bucket']:>9} {x['run']}: {x['dev_pct']:+.2f}% (stored comparison {x['stored_dev_pct']:+.2f}%), dRe {x['dre']:+.3f}")
    print(f"{len(runs)} new seeded runs: " + ', '.join(f"Q' {r['Qp']:+.0f} {r['bucket']} {r['dev_pct']:+.2f}%" for r in runs))
    print('groups of two or more: ' + ', '.join(f"Q' {p['Qp']:+.0f} {p['bucket']} ({len(p['group'])})" for p in pts if len(p['group']) > 1))


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['summary']:
        summary(); sys.exit()
    for mode in args or list(psb.buckets):
        run_bucket(mode)
    print('done')
