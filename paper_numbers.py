"""Every number that the paper quotes in Secs. III-VI, Appendix C, the tables, the figure captions and the abstract,
computed from the results in data/ (and, for the machine and simulation parameters, from the scripts that use them),
section by section. Solvers and simulations are not rerun: run the scripts first (README.md), then

    python3 paper_numbers.py

Conventions as in the paper: growth rates and tune shifts in units of Q_s; "deviation" is tracked/predicted - 1; the
"bucket effect" is the ratio of the sinusoidal bucket's growth rate to the linear theory's (most unstable mode, same rms
length; same full length for the PSB).
"""
import inspect
import json
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import ellipk, ellipj

import common as c
import seeded
import tmci
import tmci_seeds
import chromaticity as chm
import bunch_length as blm
import distributions as dsm
import damper as dam
import resistive_wall as rwm
import psb as psbm
import coupled_bunch as cbm
import linear_limit
from nonlinear import Jz_upper_from_bunch, Jz_separatrix, Jz_of_kappa

DATA = c.DATA
FRACTION = {2: 'last half', 3: 'last third', 4: 'last quarter'}          # fit windows as the paper words them


def load(name):
    with open(os.path.join(DATA, name + '.json')) as f:
        return json.load(f)


def loadv(name):
    return load(os.path.join('validation', name))


def section(title):
    print(f"\n{title}\n" + '-'*len(title))


def P(label, *values):
    print(f"  {label}: " + ', '.join(values) if values else f"  {label}")


def f(x, d=1):
    """x with d decimals, a zero printed without sign"""
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith('-') and float(s) == 0 else s


def sf(x, d=1):
    """signed, with the paper's 0.0 for a value that rounds to zero"""
    s = f"{x:+.{d}f}"
    return s[1:] if float(s) == 0 else s


def pct(x, d=1):
    return f(x, d) + '%'


def spct(x, d=1):
    return sf(x, d) + '%'


def sig(x, n=2):
    """x to n significant figures, as an integer"""
    return f"{float(f'{x:.{n}g}'):.0f}"


def sci(x, d=0):
    m, e = f"{x:.{d}e}".split('e')
    return f"{m}e{int(e)}"


def up(x, d=1):
    """x rounded up to d decimals (a bound)"""
    return f(np.ceil(x*10**d - 1e-9)/10**d, d)


def sci_up(x, d=0):
    """x rounded up to d decimals of its mantissa (a bound)"""
    e = int(np.floor(np.log10(x)))
    m = np.ceil(x/10**e*10**d - 1e-9)/10**d
    if m >= 10:
        m, e = m/10, e + 1
    return f"{m:.{d}f}e{e}"


def same_exp(lo, hi, d=1):
    """a range sharing one exponent, as the paper prints it: 'lo, hi e-n'"""
    e = int(np.floor(np.log10(hi)))
    return f(lo/10**e, d), f"{hi/10**e:.{d}f}e{e}"


def pmse(x):
    return f"{x[0]:+.2f}±{x[1]:.2f}"


def dev(q):
    """tracked/predicted - 1, in %"""
    return 100*(q['g']/q['gp'] - 1)


def devs(rows, keep=lambda q: True):
    return np.array([dev(q) for q in rows if q.get('g') is not None and q.get('gp') and keep(q)])


def rng(x, d=1):
    return spct(x.min(), d), spct(x.max(), d)


# ======================================================================================================================
tm = load('tmci'); th, tmr = tm['threshold'], tm['rows']
ch = load('chromaticity')
bl = load('bunch_length')
di = load('distributions')
da = load('damper')
rw = load('resistive_wall')
ps = load('psb')
cb = load('coupled_bunch')
ll = load('linear_limit')

# the counts of the abstract, Sec. V G and Sec. VI: every simulated growth or damping rate of the scans of Table I, the
# resistive wall at its finest slicing; a run that appears in two or three scans (the Q' = +-5 points without damper and
# with the Gaussian bunch belong to the chromaticity, damper and, in the sinusoidal bucket, distribution scans) once
rw_finest = max(q['slices'] for q in rw['tracking'])
entries = []
for rows in (tmr, ch, bl, di, da):
    entries += [q for q in rows if q.get('g') is not None and q.get('gp')]
entries += [q for q in rw['tracking'] if q['slices'] == rw_finest and q.get('g') is not None]
entries += [q for q in ps['runs'] if q.get('g') is not None and q.get('gp')]
entries += cb
seen, distinct = set(), []
for q in entries:
    k = (q['bucket'], q.get('Qp'), q['g'])
    if k not in seen:
        seen.add(k)
        distinct.append(dev(q))
distinct = np.array(distinct)
n_all, n_distinct = len(entries), len(distinct)
n_w1, n_w4 = int(np.sum(abs(distinct) < 1)), int(np.sum(abs(distinct) < 4))

# the linear limit (Fig. 1): per configuration, rows of (s, kappa_sigma, Re dQ, growth) against DELPHI's (Re dQ, growth)
S_LAW = (3, 100)          # the range of s over which the difference / kappa_sigma^2 is compared
LL = {}
for name, d in ll.items():
    re_d, g_d = d['delphi']
    rows = {r[0]: r for r in d['rows']}
    v = {s: abs(r[3]/g_d - 1)/r[1]**2 for s, r in rows.items()}               # relative difference over kappa_sigma^2
    mid = [v[s] for s in v if S_LAW[0] <= s <= S_LAW[1]]
    s_max = max(rows)
    LL[name] = dict(mid=np.sqrt(min(mid)*max(mid)), spread=max(mid)/min(mid) - 1, g300=abs(rows[s_max][3]/g_d - 1),
                    re300=abs(rows[s_max][2] - re_d), kappa300=rows[s_max][1], v300=v[s_max], vmid=mid)
S_MAX = max(max(r[0] for r in d['rows']) for d in ll.values())
assert all(max(r[0] for r in d['rows']) == S_MAX for d in ll.values())
spread = 100*max(x['spread'] for x in LL.values())
# at s = S_MAX the kappa^2 law holds where difference / kappa_sigma^2 stays inside the band it spans from s = 3 to 100
law_max = {k: min(x['vmid'])*(1 - spread/100) <= x['v300'] <= max(x['vmid'])*(1 + spread/100) for k, x in LL.items()}
g300 = sorted(x['g300'] for x in LL.values()); re300 = sorted(x['re300'] for x in LL.values())

# ======================================================================================================================
section('Sec. III (linearized Vlasov equation): the neglected chromatic term')
eps_geo = c.nemitt/(c.beta0*c.gamma)
P('rms geometric emittance (m) of the LHC beams; normalized (um)', sci(eps_geo, 1), f(c.nemitt*1e6, 1))
P("Q'_y J_y/(Q_s sigma_z) at |Q'_y| = 20: below", sci_up(20*eps_geo/(c.Qs*c.sigma_z), 0), '(1e-4)')

section('Sec. IV (matrix eigenvalue problem): the action domain')
Jup = Jz_upper_from_bunch(c.sigma_z, c.R, c.Qs, c.h, c.eta)[0]
n_sigma = inspect.signature(Jz_upper_from_bunch).parameters['n_sigma'].default
P('n_sigma; nominal LHC bunch: J_up = J_sep', f"{n_sigma:g}", str(bool(np.isclose(Jup, Jz_separatrix(c.R, c.Qs, c.h, c.eta)))))
P('(n_sigma sigma_z over the half bucket pi R/h, nominal bunch)', f(n_sigma*c.sigma_z/(np.pi*c.R/c.h), 2))

section('Appendix C (the orbit to second order in kappa)')
phi = np.linspace(0, 2*np.pi, 2001)
rem = {}
for kap in (0.2, 0.1, 0.05, 0.025):
    K = ellipk(kap**2)
    sn, cn, _, _ = ellipj(K*(1 - 2*phi/np.pi), kap**2)                  # w = K - 2K phi/pi (Sec. II C)
    z, d = 2*c.R/c.h*np.arcsin(kap*sn), 2*c.Qs*kap*cn/(c.h*abs(c.eta))   # the orbit z, delta of (J_z, phi) (Sec. II C)
    J = Jz_of_kappa(np.array([kap]), c.R, c.Qs, c.h, c.eta)[0]
    r, d_r = np.sqrt(2*J*abs(c.eta)*c.R/c.Qs), np.sqrt(2*J*c.Qs/(c.R*abs(c.eta)))      # the linear amplitudes of the same action
    rz = np.max(np.abs(z/r - np.cos(phi)*(1 + 5*kap**2/48 + kap**2/12*np.sin(phi)**2)))
    rd = np.max(np.abs(d/d_r - np.sin(phi)*(1 - kap**2/16 - kap**2/4*np.cos(phi)**2)))
    rem[kap] = (rz/kap**4, rd/kap**4)
P('remainders of the second-order orbit z, delta over kappa^4 at kappa = 0.2, 0.1, 0.05, 0.025 (constant: they fall as kappa^4)',
  *[f"{rem[k][0]:.4f}/{rem[k][1]:.4f}" for k in rem])

# ======================================================================================================================
section('Abstract')
P('linear limit at s (largest stretch): relative difference in growth rate, difference in Re dQ (Q_s), at most', str(S_MAX),
  sci_up(g300[-1]), sci_up(re300[-1]))
P('simulated growth and damping rates; within 1%; all but n within 4%', str(n_distinct), str(n_w1), '1%', str(n_distinct - n_w4), '4%')
th_shift = 100*(th['nonlinear']/th['linear'] - 1)
P('TMCI threshold lowered by', pct(-th_shift, 0))

eff_ch = {q['Qp']: q['gp'] for q in ch if q['bucket'] == 'nonlinear'}
lin_ch = {q['Qp']: q['gp'] for q in ch if q['bucket'] == 'linear'}
eff_ch = {Qp: eff_ch[Qp]/lin_ch[Qp] for Qp in eff_ch if Qp != 0}
e_bl = {(q['bucket'], q['Qp'], q['taub']): q['gp'] for q in bl}
eff_bl = {(Qp, tb): e_bl[('nonlinear', Qp, tb)]/e_bl[('linear', Qp, tb)] for (b, Qp, tb) in e_bl if b == 'nonlinear'}
e_di = {(q['bucket'], q['Qp'], q['dist']): q for q in di}
eff_di = {(Qp, ds): e_di[('nonlinear', Qp, ds)]['gp']/e_di[('linear', Qp, ds)]['gp'] for (b, Qp, ds) in e_di if b == 'nonlinear'}
e_da = {(q['bucket'], q['Qp'], q['damping']): q for q in da}
eff_da = {(Qp, dm): e_da[('nonlinear', Qp, dm)]['gp']/e_da[('linear', Qp, dm)]['gp'] for (b, Qp, dm) in e_da
          if b == 'nonlinear' and e_da[('linear', Qp, dm)]['gp']}
rs = rw['solvers']
rw_L, rw_N = dict(zip(rs['Qp'], rs['linear'])), dict(zip(rs['Qp'], rs['nonlinear']))
eff_rw = {Qp: rw_N[Qp][1]/rw_L[Qp][1] for Qp in rs['Qp'] if Qp != 0 and rw_L[Qp][1] > 0}
runs = ps['runs']
top = lambda b, Qp: max(q['gp'] for q in runs if q['bucket'] == b and q['Qp'] == Qp)
eff_ps = {Qp: top('nonlinear', Qp)/top('linear', Qp) for Qp in sorted({q['Qp'] for q in runs})}
head_tail = [*eff_ch.values(), *eff_bl.values(), *eff_di.values(), *eff_rw.values()]     # the LHC single-bunch scans without the damper
P("head-tail mode: factors from (resistive wall), to (bunch length)", f(min(head_tail), 2), f(max(head_tail), 1))

# ======================================================================================================================
section('Sec. V (introduction)')
bucket_ns = 1e9/(c.h*c.f0)
P('1-ns bunch in the bucket stretched 300 times: fraction of its length, bucket length (ns)',
  pct(100*c.taub*1e9/(300*bucket_ns), 2), sig(300*bucket_ns))
re0 = [q['rep'] for q in ch if q['Qp'] < 0]
P('mode 0 at N_b = 3e11 (Q\' < 0, both buckets): Re dQ from, to', f(max(re0), 2), f(min(re0), 2))
P('two-size criterion: Re dQ within (Q_s), growth within', f"{c.HELD_RE:g}", pct(100*c.HELD_G, 0))
sep = {tb: bool(np.isclose(Jz_upper_from_bunch(tb*1e-9*c.v/4, c.R, c.Qs, c.h, c.eta)[0], Jz_separatrix(c.R, c.Qs, c.h, c.eta)))
       for tb in np.round(blm.taub_scan*1e9, 2)}
# in a bucket stretched s times (h -> h/s at fixed Q_s), every Gaussian bunch of the paper is cut at n_sigma sigma_z: the
# stretches of Sec. V A beyond s = 1 and the 200-fold linear references, for every bunch length of the scans
stretches = sorted({r[0] for d in ll.values() for r in d['rows'] if r[0] > 1} | {200})
cut_all = all(not np.isclose(Jz_upper_from_bunch(tb*1e-9*c.v/4, c.R, c.Qs, c.h/st, c.eta)[0], Jz_separatrix(c.R, c.Qs, c.h/st, c.eta))
              for st in stretches for tb in np.round(blm.taub_scan*1e9, 2))
P('Gaussian bunches in the LHC bucket: J_up is the separatrix from tau_b (ns) on; the 6 sigma_z cut at (ns); in every stretched bucket',
  f(min(tb for tb in sep if sep[tb]), 0), *[f(tb, 1) for tb in sep if not sep[tb]], str(cut_all))
P('(the stretches checked)', *[f"{st:g}" for st in stretches])

# ======================================================================================================================
section('Table I')
npsb = {}
for q in ps['runs']:
    npsb[(q['Qp'], q['bucket'])] = npsb.get((q['Qp'], q['bucket']), 0) + 1
P('scans: N_b (1e11) from, to; Q\' from, to; tau_b (ns); damping (Q_s); resistive wall Q\', N_b; PSB Q\', modes at most; 8 bunches n_x, N_b',
  f(min(q['Nb'] for q in tmr), 2), f(max(q['Nb'] for q in tmr), 1),
  f"{min(q['Qp'] for q in ch):+.0f}", f"{max(q['Qp'] for q in ch):+.0f}", f(min(q['taub'] for q in bl), 1), f(max(q['taub'] for q in bl), 1),
  f(min(dam.damping), 0), f(max(dam.damping), 1), f"{min(rs_ := rw['solvers']['Qp']):+.0f}", f"{max(rs_):+.0f}", sci(rwm.Nb, 0),
  f"{min(q['Qp'] for q in ps['runs']):+.0f}", f"{max(q['Qp'] for q in ps['runs']):+.0f}", str(max(npsb.values())), str(len({q['nx'] for q in cb})), sci(cbm.Nb, 0))
tl, tn = devs(tmr, lambda q: q['bucket'] == 'linear'), devs(tmr, lambda q: q['bucket'] == 'nonlinear')
P('mode coupling: threshold shift; tracking linearized; sinusoidal', spct(th_shift), *rng(tl), *rng(tn))
chl = devs(ch, lambda q: q['bucket'] == 'linear' and q['Qp'] != 0); chn = devs(ch, lambda q: q['bucket'] == 'nonlinear' and q['Qp'] != 0)
P('head-tail: bucket effect; tracking linearized; sinusoidal', f(min(eff_ch.values()), 2), f(max(eff_ch.values()), 2), *rng(chl), *rng(chn))
P('bunch length: bucket effect; tracking linearized; sinusoidal', f(min(eff_bl.values()), 2), f(max(eff_bl.values()), 2),
  *rng(devs(bl, lambda q: q['bucket'] == 'linear')), *rng(devs(bl, lambda q: q['bucket'] == 'nonlinear')))
P('distributions: bucket effect; tracking linearized; sinusoidal', f(min(eff_di.values()), 2), f(max(eff_di.values()), 2),
  *rng(devs(di, lambda q: q['bucket'] == 'linear')), *rng(devs(di, lambda q: q['bucket'] == 'nonlinear')))
P('damper: bucket effect, smallest; largest; tracking linearized; sinusoidal', f(min(eff_da.values()), 0), f(max(eff_da.values()), 2),
  *rng(devs(da, lambda q: q['bucket'] == 'linear')), *rng(devs(da, lambda q: q['bucket'] == 'nonlinear')))
rw_at = {(q['Qp'], q['bucket'], q['slices']): dev(q) for q in rw['tracking'] if q.get('g') is not None}
P(f"resistive wall: bucket effect; tracking at Q' = +5, +15 with {rw_finest} slices, linearized; sinusoidal",
  f(min(eff_rw.values()), 2), f(max(eff_rw.values()), 2),
  spct(rw_at[(5., 'linear', rw_finest)]), spct(rw_at[(15., 'linear', rw_finest)]),
  spct(rw_at[(5., 'nonlinear', rw_finest)]), spct(rw_at[(15., 'nonlinear', rw_finest)]))
P('PSB: bucket effect; tracking linearized; sinusoidal', f(min(eff_ps.values()), 2), f(max(eff_ps.values()), 2),
  *rng(devs(runs, lambda q: q['bucket'] == 'linear')), *rng(devs(runs, lambda q: q['bucket'] == 'nonlinear')))
e_cb = {(q['bucket'], q['nx']): q for q in cb}
eff_cb = {nx: e_cb[('nonlinear', nx)]['gp']/e_cb[('linear', nx)]['gp'] for nx in range(8)}
P('8 bunches: bucket effect; tracking linearized; sinusoidal', f(min(eff_cb.values()), 3), f(max(eff_cb.values()), 3),
  *rng(devs(cb, lambda q: q['bucket'] == 'linear'), 2), *rng(devs(cb, lambda q: q['bucket'] == 'nonlinear'), 2))

# footnote a: the onset N_0 of g^2 = k (N_b - N_0), from the solver points within 0.3e11 above the threshold; per seed, the mean
# over the two intensities above threshold of the onset shift (g_p^2 - g^2)/k that explains the tracked rate
ts = loadv('tmci_seeds')
onset, onset_main = {}, {}
for b in ('linear', 'nonlinear'):
    pts = np.array([(q['Nb'], q['gp']**2) for q in tmr if q['bucket'] == b and q['gp'] and 0 < q['Nb'] - th[b] <= 0.3])
    k, b0 = np.polyfit(pts[:, 0], pts[:, 1], 1)
    N0 = -b0/k
    per = {}
    for r in ts['rows']:
        if r['bucket'] == b and r['g'] > 0 and r['gp'] > 0:
            per.setdefault(r['seed'], []).append(100*(r['gp']**2 - r['g']**2)/k/N0)
    d = np.array([np.mean(v) for v in per.values()])
    onset[b] = (d.mean(), d.std(ddof=1)/np.sqrt(len(d)))
    onset_main[b] = {q['Nb']: 100*(q['gp']**2 - q['g']**2)/k/N0 for q in tmr if q['bucket'] == b and q.get('g') and q['gp'] > 0}
P('footnote a: onset shift (%), linearized; sinusoidal', pmse(onset['linear']), pmse(onset['nonlinear']))
bs = loadv('bunch_length_seeded')
bsx = {(r['taub'], r['bucket'], r['mode']): r for r in bs}
gap = {tb: 100*(1 - bsx[(tb, 'linear', 'l=0 radial')]['delphi'][1]/bsx[(tb, 'linear', 'mode -1')]['delphi'][1]) for tb in (1.6, 1.7, 1.8)}
seeded_bl = {q['taub']: q for q in bl if 'g_kicked' in q}
kick_dev = {tb: 100*(q['g_kicked']/q['gp'] - 1) for tb, q in seeded_bl.items()}
P('footnote b: l = 0 radial mode within (%) of mode -1 at 1.6-1.8 ns (DELPHI); kicked runs there (%)',
  f(min(gap.values())), f(max(gap.values())), *[sf(kick_dev[tb]) for tb in (1.6, 1.7, 1.8)])
cbd = loadv('cb_delphi')['(6,48) summary']
x = max(cbd['max_abs_dre'], cbd['max_abs_dg'])
P('footnote c: DELPHI against the stretched solver, 8 bunches: tune shifts agree within (Q_s)', '1e-8' if x < 1e-8 else sci_up(x, 0))
P('(largest difference, Re dQ and growth (Q_s))', sci(cbd['max_abs_dre'], 1), sci(cbd['max_abs_dg'], 1))
dam5 = {q['damping']: dev(q) for q in da if q['bucket'] == 'nonlinear' and q['Qp'] == -5. and q.get('g') is not None}
last_grow = max(q['damping'] for q in da if q['bucket'] == 'nonlinear' and q['Qp'] == -5. and q['gp'] > seeded.GROWTH_FLOOR)
P("footnote d: mode +1 Landau damped above (Q_s, the solver); sinusoidal deviation at Q' = -5 without damper, at that damping",
  f(last_grow, 2), pct(-dam5[0.0]), pct(-dam5[last_grow], 0))
P('footnote e: slices', str(rw_finest))

# ======================================================================================================================
section('Sec. V A (linear limit)')
lc_ = {k: linear_limit.config(k) for k in ll}
P('Fig. 1: truncation (lmax, nmax), and the configurations at other than 3e11, N_b', *sorted({f"({k['lmax']},{k['nmax']})" for k in lc_.values()}),
  *[f"{n}: {k['Nb']/1e11:g}e11" for n, k in lc_.items() if k['Nb'] != 3e11])
P('largest stretch s; kappa_sigma there', str(S_MAX), sci(LL['TMCI']['kappa300'], 2))
P('s = 3 to 100: relative difference / kappa_sigma^2 constant to within', str(S_LAW[0]), str(S_LAW[1]), '10%' if spread < 10 else pct(spread, 0))
P('(largest spread over the configurations, max/min - 1)', pct(spread))
P('its value, coupled-bunch mode to merged mode at 3.8e11', f(LL['CB6']['mid'], 2), f(LL['TMCI']['mid'], 0))
P('s = 300: same law except (configurations off the band); the coupled-bunch mode\'s difference', *[k for k, ok in law_max.items() if not ok],
  sci(LL['CB6']['g300'], 1))
P('(difference / kappa_sigma^2 at s = 300 against s = 3-100, per configuration)',
  *[f"{k} {x['v300']:.3g} vs {min(x['vmid']):.3g}-{max(x['vmid']):.3g}" for k, x in LL.items()])
P('s = 300: growth rates agree to between; real tune shifts (Q_s) to between', sci(g300[0], 1), sci_up(g300[-1]), sci(re300[0], 1),
  sci_up(re300[-1]))
P('(s = 300: relative growth differences, Re dQ differences (Q_s))', *[f"{k} {x['g300']:.3g}/{x['re300']:.3g}" for k, x in LL.items()])

# ======================================================================================================================
section('Sec. V B (macroparticle simulations)')
ms = loadv('multiseed')
scatter = {}
for Qp in (5., -5.):
    for b in ('linear', 'nonlinear'):
        g = np.array([r['g'] for r in ms if r['Qp'] == Qp and r['bucket'] == b])
        gp = [r['gp'] for r in ms if r['Qp'] == Qp and r['bucket'] == b][0]
        scatter[(Qp, b)] = 100*g.std(ddof=1)/gp
P('five draws, kicked runs at Q\' = +-5: rms scatter of the growth rate from, to', f(min(scatter.values())), pct(max(scatter.values())))
nd = loadv('near_degenerate')
P('(kicked points of the bunch-length and resistive-wall scans with a second mode within 10% of the most unstable one: second/first; the three '
  'linearized ones from 1.6 ns are seeded in the scan)', *[f"{r['file']} {r['bucket']} {r['Qp']:+.0f} {r.get('taub', r.get('slices'))}: {r['second_over_first']:.3f}"
                                                         for r in nd if r['close']])
P('coordinates scaled down by 1e3 past a centroid of (m); seeded runs fitted over the; kicked runs over the; checks: at least the last (turns)',
  sci(c.RESCALE), FRACTION[seeded.FIT_SEEDED], FRACTION[seeded.FIT_KICKED], str(tmci_seeds.fit_turns))

# ======================================================================================================================
section('Sec. V C (LHC): machine and bunch')
P('C (km), h, V_rf (MV), Q_s, eta, Q_y0', f(c.circ/1e3, 2), str(c.h), f(c.V_rf/1e6, 0), sci(c.Qs, 3), sci(c.eta, 2), f(c.Qx, 2))
P('R_s (MOhm/m), f_r (GHz), Q_r, tau_b (ns)', f(c.Rs/1e6, 0), f(c.fr/1e9, 1), f(c.Qres, 0), f(c.taub*1e9, 0))
P('bunch fills of the bucket; bucket length (ns); sigma_z (cm)', pct(100*c.taub*1e9/bucket_ns, 0), f(bucket_ns, 1), f(100*c.sigma_z, 2))
mt = loadv('matching'); m0 = mt['matchings'][0]
P('sinusoidal bucket at equal sigma_z: peak line density higher by; FWHM ratio; rms momentum spread ratio',
  pct(100*(m0['peak_over_gaussian'] - 1)), f(m0['fwhm_over_gaussian'], 2), f(m0['sigma_delta']/mt['reference']['sigma_delta'], 2))
sz = {m['matching']: m['sigma_z_cm'] for m in mt['matchings']}
P('equal momentum spread: longer by, sigma_z (cm); equal emittance: longer by, sigma_z (cm)',
  pct(100*(sz['same sigma_delta']/mt['reference']['sigma_z_cm'] - 1), 0), f(sz['same sigma_delta'], 2),
  pct(100*(sz['same emittance']/mt['reference']['sigma_z_cm'] - 1), 0), f(sz['same emittance'], 2))
fac = {m['matching']: {Qp: m['growth'][Qp][1]/mt['delphi'][Qp][1] for Qp in ('5.0', '-5.0')} for m in mt['matchings']}
P("mode -1, Q' = +5: bucket effect at equal sigma_z, equal emittance, equal momentum spread",
  f(fac['same sigma_z (paper)']['5.0'], 2), f(fac['same emittance']['5.0'], 2), f(fac['same sigma_delta']['5.0'], 2))
P("mode 0, Q' = -5: bucket effect at equal sigma_z, equal emittance, equal momentum spread",
  f(fac['same sigma_z (paper)']['-5.0'], 2), f(fac['same emittance']['-5.0'], 2), f(fac['same sigma_delta']['-5.0'], 2))
# Omega_s(J_z)/omega_s = pi/(2 K(kappa^2)) on the orbit through z = n sigma_z (delta = 0): kappa = sin(h n sigma_z/2R)
Om = [np.pi/(2*ellipk(np.sin(c.h*n*c.sigma_z/(2*c.R))**2)) for n in (1, 2, 3)]
nbs = loadv('nb_scan')
r3 = next(r for r in nbs['solvers'] if r['Nb'] == 3e11)
P('Omega_s/omega_s at 1, 2, 3 sigma_z; its average over g0; rms spread', *[f(x, 3) for x in Om],
  f(r3['mean_Omegas_over_omegas'], 3), f(r3['rms_Omegas_over_omegas'], 3))
P('macroparticles; slices; normalized emittance (um)', sci(chm.num_particles), str(chm.num_slices), f(c.nemitt*1e6, 1))

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V C: mode coupling')
P('threshold (growth 0.02 Q_s), nonlinear solver; shift; DELPHI (1e11)', f(th['nonlinear'], 3), pct(-th_shift), f(th['linear'], 3))
tmm = loadv('tmci_mechanism'); dp = tmm['decomposition_points']
vv = {(x['matching'], x['variant']): x['shift_pct'] for x in tmm['variants']}
P('shift from the orbits and bunch shape; the mean synchrotron frequency; the spread about it (points)',
  f(dp['orbits_and_shape']), f(dp['mean_detuning']), sf(dp['spread']))
P('threshold shift at equal emittance (below); at equal momentum spread (above)', pct(-vv[('same emittance', 'full')]), pct(vv[('same sigma_delta', 'full')]))


def onset_of(b):
    """the first scan intensity at which the tracked run grows, and the one before it (1e11)"""
    nbs_ = sorted(q['Nb'] for q in tmr if q['bucket'] == b)
    first = min(q['Nb'] for q in tmr if q['bucket'] == b and q.get('g') is not None)            # tmci.py stores no growth below threshold
    return nbs_[nbs_.index(first) - 1], first


P('simulations become unstable between (1e11): sinusoidal; linearized', *[f(x, 2) for x in onset_of('nonlinear')], *[f(x, 2) for x in onset_of('linear')])
P('above threshold: sinusoidal runs, range; linearized runs, range', str(len(tn)), *rng(tn), str(len(tl)), *rng(tl))
q_max = max((q for q in tmr if q['bucket'] == 'linear' and q.get('g') is not None and q['gp'] > 0), key=dev)
P('largest linearized deviation, at (1e11)', spct(dev(q_max)), f(q_max['Nb'], 2))
P('sinusoidal runs: equivalent onset shift from, to (%)', f(min(onset_main['nonlinear'].values()), 2), f(max(onset_main['nonlinear'].values()), 2))
below = {b: [r for r in ts['rows'] if r['bucket'] == b and r['Nb'] == min(x['Nb'] for x in ts['rows'] if x['bucket'] == b)] for b in onset}
P('five seeds: intensity just below (1e11), runs that grow there, linearized; sinusoidal',
  f(below['linear'][0]['Nb'], 2), str(sum(r['g'] > 0 for r in below['linear'])), f(below['nonlinear'][0]['Nb'], 2), str(sum(r['g'] > 0 for r in below['nonlinear'])))
P('five seeds: tracked onset above the solver (%), linearized; sinusoidal', pmse(onset['linear']), pmse(onset['nonlinear']))
P('five seeds: turns at most; stop at centroid (cm); fitted over the last turns', str(tmci_seeds.num_turns), f(100*tmci_seeds.amp_max, 0), str(tmci_seeds.fit_turns))
first_lin = onset_of('linear')[1]
nogrow = [r for r in ts['rows'] if r['bucket'] == 'linear' and r['Nb'] == first_lin and r['g'] <= 0]
P('linearized runs at the first intensity above the threshold (1e11) that do not grow', f(first_lin, 2), str(len(nogrow)))
P('(footnote a and the onset above: the per-seed mean leaves the run that does not grow out; that seed enters through the next intensity only)',
  *[f"seed {r['seed']}" for r in nogrow])
far_rows = [q for q in tmr if q['bucket'] == 'linear' and q['Nb'] > first_lin]
P('further above threshold, from, to (1e11): linearized bucket within', f(min(q['Nb'] for q in far_rows), 1), f(max(q['Nb'] for q in far_rows), 1),
  up(max(abs(dev(q)) for q in far_rows)) + '%')
P('thresholds: solvers and simulations agree to about (the larger tracked onset shift, %)', f(max(abs(onset[b][0]) for b in onset), 1))
e_tm = {(q['bucket'], q['Nb']): q for q in tmr}
n_far = min(q['Nb'] for q in far_rows)                    # the first intensity above both thresholds, past the steep rise
P('at (1e11) the sinusoidal bucket raises the growth rate by, solvers; simulations', f(n_far, 1),
  f(e_tm[('nonlinear', n_far)]['gp']/e_tm[('linear', n_far)]['gp'], 2), f(e_tm[('nonlinear', n_far)]['g']/e_tm[('linear', n_far)]['g'], 2))
scan_nb = sorted({q['Nb'] for q in tmr})
n_below = scan_nb[scan_nb.index(onset_of('nonlinear')[0]) - 1]     # the intensity before the last stable one of the sinusoidal bucket
re_below = max(abs(q['re'] - q['rep']) for q in tmr if q['Nb'] <= n_below and q.get('re') is not None and q.get('rep') is not None)
P('below threshold, up to (1e11): mode 0 tune shifts agree within (Q_s)', f(n_below, 2), up(re_below, 3))
n_last = onset_of('nonlinear')[0]
P('(at the last intensity below the sinusoidal threshold (1e11): |Re dQ tracked - predicted| (Q_s), linearized; sinusoidal)', f(n_last, 2),
  *[f"{abs(e_tm[(b, n_last)]['re'] - e_tm[(b, n_last)]['rep']):.4f}" for b in ('linear', 'nonlinear')])
P('the sinusoidal bucket shifts mode 0 faster at 3e11 by', pct(100*(e_tm[('nonlinear', 3.0)]['rep']/e_tm[('linear', 3.0)]['rep'] - 1), 0))
P('truncation (lmax, nmax); e-foldings; at most turns; turns below threshold', f"({tmci.lmax},{tmci.nmax})", str(seeded.EFOLDINGS), str(tmci.max_turns),
  str(seeded.STABLE_TURNS))

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V C: chromaticity')
P("Q' < 0 (mode 0): bucket effect at Q' = -10; at Q' = -1", pct(100*(eff_ch[-10.] - 1), 0), pct(100*(eff_ch[-1.] - 1), 0))
P("Q' > 0 (mode -1): bucket effect from, to", f(min(v for k, v in eff_ch.items() if k > 0), 2), f(max(v for k, v in eff_ch.items() if k > 0), 2))
P('sinusoidal tracking: range; runs; rms', *rng(chn), str(len(chn)), pct(np.sqrt(np.mean(chn**2))))
P("linearized tracking: Q' < 0; Q' > 0", *rng(devs(ch, lambda q: q['bucket'] == 'linear' and q['Qp'] < 0)), *rng(devs(ch, lambda q: q['bucket'] == 'linear' and q['Qp'] > 0)))
dl_pos = {q['Qp']: dev(q) for q in ch if q['bucket'] == 'linear' and q['Qp'] > 0}
top = sorted(dl_pos, key=dl_pos.get)[len(dl_pos)//2:]                  # the larger half of the deviations
assert sorted(top) == [Qp for Qp in sorted(dl_pos) if Qp >= min(top)]   # they are the larger Q'
P("the larger half of the linearized deviations at Q' > 0 lie at Q' >=", f"{min(top):+.0f}")
P("(linearized deviations at Q' = +1 ... +10, %)", *[f(dev(q), 2) for q in sorted(ch, key=lambda q: q['Qp']) if q['bucket'] == 'linear' and q['Qp'] > 0])
lc = loadv('linear_control')
P("Q' = +8: kicked at 4x particles and 2x slices; kicked at the scan's resolution; seeded", spct(lc['resolution']['dev']), spct(lc['repro']['dev']), spct(lc['seeded']['dev_fit']))
P("(Q' = +8, linearized: the scan's own seeded run)", spct(next(dev(q) for q in ch if q['bucket'] == 'linear' and q['Qp'] == 8.)))
P('real tune shifts agree within (Q_s)', up(max(abs(q['re'] - q['rep']) for q in ch if q.get('g') is not None), 3))
cbv = loadv('convergence_band')
c0 = next(x for x in cbv if x['scan'] == 'chromaticity' and x['Qp'] == 0. and x['bucket'] == 'nonlinear')
q0 = {q['bucket']: q for q in ch if q['Qp'] == 0.}
P("Q' = 0: largest predicted growth (Q_s), sinusoidal at (6,32), absent at n_max = 48, 64; kicked runs of ... turns grow", sci(q0['nonlinear']['gp'], 1),
  *[str(n) for n in c0['sizes'][1:]], sci(seeded.KICKED_TURNS), str(any(q0[b]['g'] is not None for b in q0)))
P("(Q' = 0: linearized prediction; sinusoidal fastest unstable eigenvalue at n_max = 48, 64)", sci(q0['linear']['gp'], 1),
  sci(c0['fastest']['48'][1], 0), sci(c0['fastest']['64'][1], 0))
P('seed threshold: no seed below a predicted growth of (Q_s); truncation of the solvers; of the seed', sci(seeded.GROWTH_FLOOR),
  f"({chm.lmax},{chm.nmax})", f"({chm.lmax},{chm.nmax_seed})")

# the intensity scan at Q' = +5 (nb_scan.json, convergence_band.json, nb_l0_seeded.json)
s11 = next(r for r in nbs['solvers'] if r['Nb'] == 1e11); s310 = next(r for r in nbs['solvers'] if r['Nb'] == 3e10)
c11 = next(x for x in cbv if x['scan'] == 'nb_scan' and x['Nb'] == 1e11 and x['which'] == 'nonlinear_mode_m1')
f11 = {int(n): v[1] for n, v in c11['fastest'].items()}
g_lin11 = s11['delphi_mode_m1'][1]
P('1e11: linear theory, mode -1', sci(g_lin11, 1))
P('(1e11: fastest l = -1 eigenvalue for n_max = ' + ', '.join(map(str, sorted(f11))) + ')', *[sci(f11[n], 2) for n in sorted(f11)])
P('1e11: n_max from, to; the fastest l = -1 eigenvalue between; relative to the linear value between', str(min(f11)), str(max(f11)),
  *same_exp(min(f11.values()), max(f11.values())), f(min(f11.values())/g_lin11, 1), f(max(f11.values())/g_lin11, 1))
nbt = {(r['Nb'], r['bucket'], r['run']): r for r in nbs['tracking']}
s1 = nbt[(1e11, 'nonlinear', 'seeded l=-1')]
gfits = [s1['g_pencil'], s1['g_early'], s1['g']]
P('1e11, seeded l = -1 pattern: growth from, to (fits); relative to the linear value', *same_exp(min(gfits), max(gfits)),
  f(min(gfits)/g_lin11, 2), f(max(gfits)/g_lin11, 2))
c310 = next(x for x in cbv if x['scan'] == 'nb_scan' and x['Nb'] == 3e10)
w310 = {int(n): max(x[1] for x in v) for n, v in c310['window_m1'].items()}
P('3e10: linear theory, mode -1; the l = -1 eigenvalues fall with n_max; the seeded pattern decays', sci(s310['delphi_mode_m1'][1], 1),
  str(all(w310[a] > w310[b] for a, b in zip(sorted(w310), sorted(w310)[1:]))), str(nbt[(3e10, 'nonlinear', 'seeded l=-1')]['g'] < 0))
P('(3e10: fastest l = -1 eigenvalue for n_max = 32 ... 96; growth of the seeded run)', *[sci(w310[n], 1) for n in sorted(w310)],
  sci(nbt[(3e10, 'nonlinear', 'seeded l=-1')]['g'], 1))
nl0 = {r['bucket']: r for r in loadv('nb_l0_seeded')}
P('(the fastest mode at 1e11 and 3e10 is an l = 0 radial mode: Re dQ (Q_s) of the most unstable mode, DELPHI; nonlinear solver)',
  *[f"{x:+.3f}" for s_ in (s11, s310) for x in (s_['delphi'][0], s_['nonlinear_most'][0])])
P('l = 0 radial mode, bucket effect: solvers (1e11, 3e10); seeded simulations (1e11); kicked simulations (3e10)',
  pct(100*(s11['nonlinear_most'][1]/s11['delphi'][1] - 1)), pct(100*(s310['nonlinear_most'][1]/s310['delphi'][1] - 1)),
  pct(100*(nl0['nonlinear']['g']/nl0['linear']['g'] - 1)), pct(100*(nbt[(3e10, 'nonlinear', 'kicked')]['g']/nbt[(3e10, 'linear', 'kicked')]['g'] - 1)))
re5 = {q['bucket']: q['rep'] for q in ch if q['Qp'] == 5.}
P("Q' = +5, 3e11: Re dQ of mode -1, linearized to sinusoidal", f(re5['linear'], 3), f(re5['nonlinear'], 3))
g_d, g_n, g_ws, g_mean = r3['delphi_mode_m1'][1], r3['nonlinear_mode_m1'][1], r3['constant_Omegas_mode_m1'][1], r3['constant_meanOmegas_mode_m1'][1]
P('constant Omega_s = omega_s: growth; share of the way (in log) from DELPHI to the nonlinear solver; DELPHI; nonlinear solver',
  f(g_ws, 4), pct(100*np.log(g_ws/g_d)/np.log(g_n/g_d), 0), f(g_d, 4), f(g_n, 4))
P('constant Omega_s = its average over g0: ratio; growth; share of the way', f(r3['mean_Omegas_over_omegas'], 3), f(g_mean, 4),
  pct(100*np.log(g_mean/g_d)/np.log(g_n/g_d), 0))
P('factor at Q\' = +5 = orbits and shape x mean synchrotron frequency x spread', f(g_n/g_d, 2), f(g_ws/g_d, 2), f(g_mean/g_ws, 2), f(g_n/g_mean, 2))

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V C: bunch length')
P('tau_b from, to (ns); fraction of the bucket from, to', f(min(q['taub'] for q in bl), 1), f(max(q['taub'] for q in bl), 1),
  pct(100*min(q['taub'] for q in bl)/bucket_ns, 0), pct(100*max(q['taub'] for q in bl)/bucket_ns, 0))
eff_p = {tb: v for (Qp, tb), v in eff_bl.items() if Qp == 5.}
eff_m = {tb: v for (Qp, tb), v in eff_bl.items() if Qp == -5.}
P("Q' = +5: bucket effect at 0.6 ns; largest, at (ns)", pct(100*(eff_p[0.6] - 1), 0), f(max(eff_p.values()), 2), f(max(eff_p, key=eff_p.get), 1))
l0 = loadv('l0_share')
rad = [r for r in l0 if r['mode'] == 'l=0 radial' and r['bucket'] == 'nonlinear']
P('beyond 1.5 ns the most unstable mode is an l = 0 radial mode: l = 0 content (least); Re dQ',
  pct(100*min(min(r['l0_share_coefficients'], r['l0_share_weighted']) for r in rad), 0),
  sf(np.mean([bsx[(tb, 'nonlinear', 'l=0 radial')]['solver'][0] for tb in (1.6, 1.7, 1.8)]), 2))
nl_rep = {q['taub']: q['rep'] for q in bl if q['bucket'] == 'nonlinear' and q['Qp'] == 5.}
P("Fig. 4(a): the most unstable mode of the sinusoidal bucket changes from mode -1 (Re dQ near -1) to the l = 0 mode between (ns)",
  f(max(tb for tb, x in nl_rep.items() if x < -0.5), 1), f(min(tb for tb, x in nl_rep.items() if abs(x) < 0.5), 1))
r_l0 = {tb: bsx[(tb, 'nonlinear', 'l=0 radial')]['solver'][1]/bsx[(tb, 'linear', 'l=0 radial')]['delphi'][1] for tb in (1.6, 1.7, 1.8)}
P('l = 0 radial mode, sinusoidal over linearized (1.6-1.8 ns), from, to', f(min(r_l0.values()), 2), f(max(r_l0.values()), 2))
r_m1 = {tb: bsx[(tb, 'nonlinear', 'mode -1')]['solver'][1]/bsx[(tb, 'linear', 'mode -1')]['delphi'][1] for tb in (1.6, 1.7, 1.8)}
held_m1 = {tb: bsx[(tb, 'nonlinear', 'mode -1')]['held'] for tb in (1.6, 1.7, 1.8)}
nl16 = bsx[(1.6, 'nonlinear', 'mode -1')]
P('mode -1, sinusoidal solver over DELPHI at 1.6, 1.7 ns; seeded run at 1.6 ns against the solver; held at 1.8 ns',
  '1.6', f(r_m1[1.6], 2), '1.7', f(r_m1[1.7], 2), pct(abs(100*(nl16['g_fit']/nl16['solver'][1] - 1))), str(held_m1[1.8]))
nl17 = bsx[(1.7, 'nonlinear', 'mode -1')]
P('  seeded mode -1 at 1.7 ns, sinusoidal: fit on the seeded mode', str(nl17['fit_on_mode']))
long_ = {tb: v for tb, v in eff_p.items() if tb > 1.5}
P('ratio of the largest growth rates beyond 1.5 ns, from, to', f(min(long_.values()), 2), f(max(long_.values()), 2))
P("Q' = -5: bucket effect from, to", pct(100*(min(eff_m.values()) - 1)), pct(100*(max(eff_m.values()) - 1), 0))
q15 = next(r for r in nd if r['file'] == 'bunch_length' and r['bucket'] == 'linear' and r['Qp'] == 5. and r['taub'] == 1.5)
P('linearized bucket, l = 0 mode slower than mode -1 at 1.5 ns; at 1.6, 1.7, 1.8 ns (DELPHI)',
  pct(100*(1 - q15['second_over_first'])), *[f(gap[tb]) for tb in (1.6, 1.7, 1.8)])
P('kicked linearized runs against mode -1 at 1.5 ns (within); 1.6; 1.7; 1.8 ns', up(abs(q15['dev'])) + '%', *[spct(kick_dev[tb]) for tb in (1.6, 1.7, 1.8)])
q18 = seeded_bl[1.8]
P('at 1.8 ns: kicked run at the frequency of; below the l = 0 mode by',
  'l=0' if abs(q18['re_kicked'] - bsx[(1.8, 'linear', 'l=0 radial')]['solver'][0]) < 0.25 else 'mode -1',
  pct(-100*(q18['g_kicked']/bsx[(1.8, 'linear', 'l=0 radial')]['delphi'][1] - 1)))
sm1 = [100*(bsx[(tb, 'linear', 'mode -1')]['g_fit']/bsx[(tb, 'linear', 'mode -1')]['delphi'][1] - 1) for tb in (1.6, 1.7, 1.8)]
sl0 = [100*(bsx[(tb, 'linear', 'l=0 radial')]['g_fit']/bsx[(tb, 'linear', 'l=0 radial')]['delphi'][1] - 1) for tb in (1.6, 1.7, 1.8)]
P('seeded linearized runs against DELPHI: mode -1 from, to; l = 0 mode from, to', sf(min(sm1)), spct(max(sm1)), sf(min(sl0)), spct(max(sl0)))
P('  (the seeded mode -1 runs are the points of Fig. 4(a): stored growth equals the seeded run)',
  str(all(seeded_bl[tb]['g'] == bsx[(tb, 'linear', 'mode -1')]['g_fit'] for tb in (1.6, 1.7, 1.8))))
grow = [q for q in bl if q.get('g') is not None and q.get('gp')]
on_mode = [q for q in grow if q.get('rep') is None or abs(q['re'] - q['rep']) <= 0.02]
w1 = [q for q in grow if abs(dev(q)) < 1]
rest = sorted((q for q in grow if abs(dev(q)) >= 1), key=lambda q: q['taub'])
P('simulations; on the predicted mode; within 1%', str(len(grow)), str(len(on_mode)), str(len(w1)))
P('the others: bucket, Q\', tau_b (ns), predicted growth, deviation',
  *[f"{q['bucket']} {q['Qp']:+.0f} {q['taub']} {f(q['gp'], 3)} {pct(-dev(q), 2)} low" for q in rest])
P('kicked runs: e-foldings; linearized slicer at least (sigma_z); basis: lmax, nmax, nmax_check', str(blm.efoldings), str(blm.slicer_nsigma),
  str(blm.lmax), str(blm.nmax), str(blm.nmax_check))

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V C: longitudinal distribution (and Table II)')
P('sigma_z (cm)', f(100*c.sigma_z, 2))
P('edges kappa_e: waterbag, parabolic line, parabolic amplitude',
  *[f(dsm.bucket(p, c.h)['kappa_edge'], 3) for name, p in dsm.distributions if p is not None])
P("Q' = +5: bucket effect, waterbag; parabolic amplitude", f(eff_di[(5., 'waterbag')], 2), f(eff_di[(5., 'parabolic amplitude')], 2))
m5 = [v for (Qp, ds), v in eff_di.items() if Qp == -5.]
P("Q' = -5: bucket effect, all four", pct(100*(min(m5) - 1), 0), pct(100*(max(m5) - 1), 0))
dd = devs(di)
P('simulations; within; e-foldings; solver truncation, checked against', str(len(dd)), pct(abs(dd).max()), str(seeded.EFOLDINGS),
  f"({dsm.lmax},{dsm.nmax_check})", f"({dsm.lmax},{dsm.nmax})")
for Qp in (5., -5.):
    for name in ('waterbag', 'parabolic line', 'Gaussian', 'parabolic amplitude'):             # the order of Table II
        q = {b: e_di[(b, Qp, name)] for b in ('linear', 'nonlinear')}
        P(f"Table II, Q' = {Qp:+.0f}, {name}: linearized solver, tracking; sinusoidal solver, tracking (1e-2)",
          *[f(100*q[b][k], 2) for b in ('linear', 'nonlinear') for k in ('gp', 'g')])

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V C: transverse damper')
P('damping from, to (Q_s)', f(min(dam.damping), 0), f(max(dam.damping), 1))
eff_p5 = {dm: v for (Qp, dm), v in eff_da.items() if Qp == 5.}
top_d = sorted(dm for dm, v in eff_p5.items() if f(v, 2) == f(max(eff_p5.values()), 2))
P("Q' = +5: bucket effect without damper; largest; at damping (Q_s)", f(eff_p5[0.0], 2), f(max(eff_p5.values()), 2), *[f"{x:g}" for x in top_d])
P("removed by the damper at 0.125 Q_s, linearized; sinusoidal (Q_s)",
  f(e_da[('linear', 5., 0.0)]['gp'] - e_da[('linear', 5., 0.125)]['gp'], 3), f(e_da[('nonlinear', 5., 0.0)]['gp'] - e_da[('nonlinear', 5., 0.125)]['gp'], 3))
lm5 = sorted((q for q in da if q['bucket'] == 'linear' and q['Qp'] == -5.), key=lambda q: q['damping'])
q1 = next(q for q in lm5 if q['rep'] is not None and abs(q['rep'] - 1) < 0.25)          # the first damping whose most unstable mode is mode +1
P("Q' = -5, linearized: mode +1 (Re dQ near +1) takes over from (Q_s), at growth (Q_s)", f(q1['damping'], 2), f(q1['gp'], 3))
P("(the most unstable mode at the damping before, Re dQ)", f(lm5[lm5.index(q1) - 1]['rep'], 2))
dl_ = loadv('damper_long')
P("mode +1 there: Re dQ (Q_s), inside the l = +1 band (0 to 1)", sf(q1['rep'], 2))
P("(predicted and tracked growth from there on, linearized (Q_s))", *[f"{q['damping']:g}: {q['gp']:.4f}/{q['g']:.4f}" for q in lm5 if q['damping'] >= q1['damping']])
P('rms spread of Omega_s (omega_s)', f(r3['rms_Omegas_over_omegas'], 3))
held_above = [v['held'][1] for k, v in dl_['solvers'].items() if float(k) > 0.15 and v['held']]
nogrow_gp = [q['gp'] for q in da if q['bucket'] == 'nonlinear' and q['Qp'] == -5. and q['damping'] > 0.15]
P('above 0.15 Q_s, sinusoidal: no held mode growing faster than (Q_s); band eigenvalues not held at (Q_s)',
  sci_up(max(held_above + nogrow_gp), 0), sci(np.mean([v['fastest_any'][1] for k, v in dl_['solvers'].items() if float(k) > 0.15]), 1))
nog = [q for q in da if q['bucket'] == 'nonlinear' and q['Qp'] == -5. and q['damping'] > 0.15]
P('simulations above 0.15 Q_s, sinusoidal; that grow', str(len(nog)), str(sum(q.get('g') is not None for q in nog)))
lt = [r for r in dl_['tracking'] if r['damping'] > 0.15]
P('long runs: turns; at damping (Q_s); band eigenvalue e-foldings in such a run (least); growth resolved above (Q_s); any grows',
  sci(min(r['turns'] for r in lt), 0), *[f"{r['damping']:g}" for r in lt],
  f(min(2*np.pi*c.Qs*r['turns']*dl_['solvers'][f"{r['damping']}"]['fastest_any'][1] for r in lt), 0),
  sci(max(r['resolvable_growth'] for r in lt), 1), str(any(r['grew'] for r in lt)))
grow_da = [q for q in da if q.get('g') is not None and q.get('gp')]
dd = np.array([dev(q) for q in grow_da])
mid = [q for q in grow_da if 1 <= abs(dev(q)) < 4]
worst = max(grow_da, key=lambda q: abs(dev(q)))
P('simulations that grow; within 1%; within 1-4%, all at damping of at least (Q_s)', str(len(grow_da)), str(int(np.sum(abs(dd) < 1))),
  str(len(mid)), f(min(q['damping'] for q in mid), 1))
P('the remaining one: deviation (low); bucket, Q\', damping; tracked; predicted', pct(-dev(worst)), worst['bucket'], f"{worst['Qp']:+.0f}",
  f(worst['damping'], 2), f(worst['g'], 4), f(worst['gp'], 4))
lin5 = {q['damping']: dev(q) for q in da if q['bucket'] == 'linear' and q['Qp'] == -5. and q.get('g') is not None}
P("Q' = -5 below the solvers: without damper (both buckets, below); 0.10 Q_s sinusoidal (linearized); 0.125 Q_s sinusoidal (linearized)",
  up(max(abs(dam5[0.0]), abs(lin5[0.0]))) + '%', '0.10', pct(-dam5[0.1]), pct(-lin5[0.1]), pct(-dam5[0.125]), pct(-lin5[0.125]))
gp5 = {q['damping']: q['gp'] for q in da if q['bucket'] == 'nonlinear' and q['Qp'] == -5.}
absdev = {dm: gp5[dm] - (dev(e_da[('nonlinear', -5., dm)])/100 + 1)*gp5[dm] for dm in dam5}
P("mode 0 growth rate without damper, at 0.15 Q_s; absolute deviation without damper, at 0.15 Q_s (Q_s)",
  f(gp5[0.0], 2), f(gp5[0.15], 3), sci(absdev[0.0], 0), sci(absdev[0.15], 1))
dtr = loadv('damper_trend')
res = {r['damping']: r for r in dtr if r['test'] == 'resolution' and r['Qp'] == -5.}
P('4x particles, 2x slices (kicked, sinusoidal) reproduce the deviations within (points)', up(max(abs(res[d]['dev'] - dam5[d]) for d in res), 2))
band = [x for x in cbv if x['scan'] == 'damper' and x['Qp'] == -5. and x['damping'] <= 0.15]
band_max = 100*max(abs(v[1]/x['stored'][1] - 1) for x in band for n, v in x['fastest'].items() if v is not None and x['stored'][1] > 0)
P('the predictions change between n_max = 32, 48 and 64 by less than', *[str(n) for n in band[0]['sizes']], up(band_max, 0) + '%')
P('solver truncation of the damper scan (as the bunch-length scan): nonlinear, checked against; DELPHI', f"({dam.lmax},{dam.nmax_check})",
  f"({dam.lmax},{dam.nmax})", f"({dam.lmax},{dam.nmax})")
P('(largest change, nonlinear solver; DELPHI (6,32) to (6,48) and (6,64), in %)', pct(band_max, 2), sci(loadv('delphi_damper_basis')['max_change_pct'], 0))
cal = [r for r in dtr if r['test'] == 'calibration']
P('(rigid bunch, no wake: decay rate over the set rate, against 1 + d, largest difference)',
  sci(max(abs(r['ratio'] - (1 + 2*np.pi*c.Qs*r['damping'])) for r in cal), 0))
excess = {d: 100*(2*np.pi*c.Qs*d)*d/gp5[d] for d in (0.10, 0.125, 0.15)}
P('excess damping a d (a = 2 pi Q_s d) over the predicted growth rate, sinusoidal, at 0.10, 0.125, 0.15 Q_s', *[pct(excess[d]) for d in (0.10, 0.125, 0.15)])
dlu = loadv('damper_lumped')
dlv = {(r['variant'], r['bucket'], r['damping']): r['dev_pct'] for r in dlu if r['test'] == 'seeded'}
P('damper before the wake, 2 and 4 kick points: linearized at 0.125 Q_s (from the scan\'s)', spct(lin5[0.125]),
  *[spct(dlv[(v, 'linear', 0.125)]) for v in ('B', 'N2', 'N4')])
P('  ... sinusoidal at 0.15 Q_s (from the scan\'s)', spct(dam5[0.15]), *[spct(dlv[(v, 'nonlinear', 0.15)]) for v in ('B', 'N2', 'N4')])
sh0 = [dlv[(v, b, 0.0)] - (dam5[0.0] if b == 'nonlinear' else lin5[0.0]) for v in ('N2', 'N4') for b in ('linear', 'nonlinear')]
P('spreading the kicks moves the points without damper by from, to (points)', f(min(sh0)), pct(max(sh0)))
n4 = [dlv[('N4', 'nonlinear', d)] for d in (0.0, 0.10, 0.125, 0.15)]
P('four kick points, sinusoidal, 0 to 0.15 Q_s: from, to', spct(n4[0]), spct(n4[-1]))
ctl = [r for r in dlu if r['test'] == 'seeded' and r['variant'] == 'A']
P('  (the A control reproduces the scan bit for bit)', str(all(r['identical_to_stored_g'] for r in ctl)))
P('kicked where no mode is predicted to grow: turns', sci(seeded.KICKED_TURNS))

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V D: resistive wall')
P('pipe radius (cm); resistivity (Ohm m); N_b', f(100*rwm.radius, 0), sci(rwm.rho, 1), sci(rwm.Nb, 0))
P("mode -1 at Q' = +5: change; DELPHI; nonlinear solver (1e-2)", pct(-100*(eff_rw[5.] - 1), 0), f(100*rw_L[5.][1], 3), f(100*rw_N[5.][1], 3))
P("mode -1 at Q' = +15: change; DELPHI; nonlinear solver (1e-2)", pct(100*(eff_rw[15.] - 1), 0), f(100*rw_L[15.][1], 3), f(100*rw_N[15.][1], 3))
m1 = lambda sol, Qp: sol[Qp][1] if Qp > 0 and abs(sol[Qp][0] + 1) < 0.5 else 0.          # mode -1: Q' > 0, Re dQ near -1
P("mode -1 growth rate peaks at Q' (DELPHI; nonlinear solver)", f"{max(rs['Qp'], key=lambda q: m1(rw_L, q)):.0f}", f"{max(rs['Qp'], key=lambda q: m1(rw_N, q)):.0f}")
P("Q' < 0: the solvers agree within", pct(100*max(abs(v - 1) for k, v in eff_rw.items() if k < 0)))
P("Q' > 0: ratio at Q' = +1; at +16", f(eff_rw[1.], 2), f(eff_rw[16.], 2))
P("largest ratio, at Q'", f(max(eff_rw.values()), 2), f"{max(eff_rw, key=eff_rw.get):+.0f}")
mode_m2 = [Qp for Qp in rs['Qp'] if Qp > 16 and abs(rw_L[Qp][0] + 2) < 0.5]
P("above Q' = +16 DELPHI's most unstable mode is mode -2 (Re dQ near -2), from Q'", f"{min(mode_m2):+.0f}")
P('truncation: DELPHI; nonlinear solver, checked against; growth at Q\' = 0 below (Q_s)', f"({rwm.lmax},{rwm.nmax})", f"({rwm.lmax},{rwm.nmax_check})",
  f"({rwm.lmax},{rwm.nmax})", f"1e{int(np.ceil(np.log10(max(rw_L[0.][1], rw_N[0.][1]))))}")
P("(growth rates at Q' = 0, DELPHI; nonlinear solver (Q_s))", sci(rw_L[0.][1], 1), sci(rw_N[0.][1], 1))
P('macroparticles at most; turns of wake memory, at 12800-25600 slices and below; e-foldings of the slower bucket; fit over the last',
  sci(max(rwm.settings(N)[0] for N in (25600,)), 0), str(rwm.settings(25600)[1]), str(rwm.settings(400)[1]), f(rwm.efoldings, 0),
  'third' if rwm.fit_fraction == 3 else f"1/{rwm.fit_fraction}")
P("Q' = +5, sinusoidal, above the solver with 400; 6400; 25600 slices", '400', pct(rw_at[(5., 'nonlinear', 400)], 0), '6400',
  pct(rw_at[(5., 'nonlinear', 6400)], 0), '25600', pct(rw_at[(5., 'nonlinear', 25600)]))
P("25600 slices: linearized +5 (below); +15 (above); sinusoidal +5; +15 (above)", pct(-rw_at[(5., 'linear', 25600)]), pct(rw_at[(15., 'linear', 25600)]),
  pct(rw_at[(5., 'nonlinear', 25600)]), pct(rw_at[(15., 'nonlinear', 25600)]))
# the series of the linearized bucket at Q' = +5 has settled from the smallest slicing whose deviations from there on all lie
# within 0.5% of DELPHI
s5 = sorted(N for N in rwm.slices[5.] if (5., 'linear', N) in rw_at)
N_set = next(N for N in s5 if all(abs(rw_at[(5., 'linear', M)]) < 0.5 for M in s5 if M >= N))
settled = max(abs(rw_at[(5., 'linear', N)]) for N in s5 if N >= N_set)
P("linearized Q' = +5 settled within (nearest), from (slices)", pct(settled), str(N_set))
P("(linearized Q' = +5: deviations from there on, %)", *[f"{N}: {rw_at[(5., 'linear', N)]:+.3f}" for N in s5 if N >= N_set])


def effect_tracked(N):
    g = {(q['Qp'], q['bucket']): q['g'] for q in rw['tracking'] if q['slices'] == N}
    return {Qp: 100*(g[(Qp, 'nonlinear')]/g[(Qp, 'linear')] - 1) for Qp in (5., 15.)}


et = effect_tracked(25600)
P("tracked bucket effect at 25600 slices, Q' = +5; +15; solvers", pct(et[5.], 0), spct(et[15.], 0), pct(100*(eff_rw[5.] - 1), 0), spct(100*(eff_rw[15.] - 1), 0))
et12 = effect_tracked(12800)
P('tracked bucket effect at 12800 slices', '12800', pct(et12[5.], 0), spct(et12[15.], 0))
fits = {}
for Qp in (5., 15.):
    for b in ('nonlinear', 'linear'):
        rr = sorted((q for q in rw['tracking'] if q['Qp'] == Qp and q['bucket'] == b and q.get('g') is not None), key=lambda q: q['slices'])
        N = np.array([q['slices'] for q in rr], float); y = np.array([q['g']/q['gp'] for q in rr])
        (a, b_, p), _ = curve_fit(lambda n, a, b, p: a + b*n**(-p), N, y, p0=(1., 5., 0.5), maxfev=20000)
        fits[(Qp, b)] = (a, p)
conv = [k for k in fits if k != (5., 'linear')]
P('fit a + b N^-p to the three converging series: p from, to; a from, to', f(min(fits[k][1] for k in conv), 2), f(max(fits[k][1] for k in conv), 2),
  f(min(fits[k][0] for k in conv), 3), f(max(fits[k][0] for k in conv), 3))
a_lin5 = next(q['g']/q['gp'] for q in rw['tracking'] if q['Qp'] == 5. and q['bucket'] == 'linear' and q['slices'] == 25600)
e5 = 100*(eff_rw[5.]*fits[(5., 'nonlinear')][0]/a_lin5 - 1)
e15 = 100*(eff_rw[15.]*fits[(15., 'nonlinear')][0]/fits[(15., 'linear')][0] - 1)
P('settled series at its measured; extrapolated tracked bucket effects', f(a_lin5, 3), pct(e5, 0), spct(e15, 0))

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V E: PSB')
P('kinetic energy (MeV); eta; h; V_rf (kV); Q_s; Q_y0', f(psbm.Ekin/1e6, 0), f(psbm.eta, 3), str(psbm.h), f(psbm.V_rf/1e3, 0), sci(psbm.Qs, 3), f(psbm.Qx, 2))
P('N_b; full length (ns); fraction of the bucket; kappa_e', sci(psbm.Nb, 0), f(psbm.taub*1e9, 0), 'half' if abs(psbm.taub*psbm.f0/psbm.h - 0.5) < 0.01 else f(psbm.taub*psbm.f0/psbm.h, 2),
  f(psbm.kappa_edge, 3))
P("R_s (MOhm/m); f_r (MHz); Q_r; Q' from, to", f(psbm.Rt/1e6, 0), f(psbm.fr/1e6, 2), f(psbm.Qres, 0), f"{psbm.Qpscan[0]:+.0f}", f"{psbm.Qpscan[-1]:+.0f}")
P('truncation: lmax; nmax, nmax_check (nonlinear); DELPHI nmax', str(psbm.lmax), str(psbm.nmax), str(psbm.nmax_check), str(psbm.nmax_delphi))
# rms lengths of the two bunches of the same full length: parabolic line density in sin(hz/2R) (sinusoidal) and in z (linear)
zz = np.linspace(-psbm.z_b, psbm.z_b, 400001)
lam_lin = 1 - (zz/psbm.z_b)**2
s_ = np.sin(psbm.h*zz/(2*psbm.R))/psbm.kappa_edge
lam_sin = np.clip(1 - s_**2, 0, None)              # g0 = (1 - kappa^2/kappa_e^2)^(1/2) integrated over delta
rms = lambda lam: np.sqrt(np.trapezoid(lam*zz**2, zz)/np.trapezoid(lam, zz))
P('rms lengths of the two bunches differ by', pct(100*abs(rms(lam_sin)/rms(lam_lin) - 1)))
dx = loadv('psb_delphi_x86') if os.path.exists(os.path.join(DATA, 'validation', 'psb_delphi_x86.json')) else None
dd_ = {r['Qp']: r for r in ps['delphi']}
top_line = lambda Qp: max(q['gp_line'] for q in runs if q['bucket'] == 'linear' and q['Qp'] == Qp)
dlp = {Qp: 100*(dd_[Qp]['g']/top_line(Qp) - 1) for Qp in dd_}
n_lag = len(c.DELPHI.longdistribution_decomp(psbm.taub, typelong='parabolicline')[0])
P("DELPHI's parabola (Laguerre terms) against the stretched solver at the lines: within, down to Q' = -6; lower at -7, -8", str(n_lag),
  pct(max(abs(dlp[q]) for q in dlp if q >= -6)), pct(-dlp[-7.], 0), pct(-dlp[-8.], 0))
P("(largest difference for Q' >= -2, in %)", f"{max(abs(dlp[q]) for q in dlp if q >= -2):.1e}")
if dx is None:                     # psb_delphi_x86.py is machine-specific and not part of run_all.sh
    P('(DELPHI on another machine at -7, -8: not available, run psb_delphi_x86.py)')
else:
    dxr = {r['Qp']: r for r in dx['rows']}
    P('(DELPHI on another machine at -7, -8: relative growth difference)', *[pct(100*abs(dxr[Qp]['g_rel_diff']), 0) for Qp in (-7., -8.)])
P('macroparticles; slices; turns of wake memory; e-foldings', sci(psbm.num_particles, 0), str(psbm.num_slices), str(psbm.num_turns_wake), str(psbm.efoldings))
gmax = max(max(q['gp'], q['gp_line']) for q in runs)
P('growth rates reach (Q_s); in kHz; resonance half-width (kHz)', f(gmax, 1), f(gmax*psbm.Qs*psbm.f0/1e3, 0), f(psbm.fr/(2*psbm.Qres)/1e3, 0))
P('impedance at the modes\' frequencies changes the growth rates by up to', pct(100*max(abs(q['gp']/q['gp_line'] - 1) for q in runs if q['gp_line']), 0))
pq = loadv('psb_q100')
P('Q_r = 100: turns of wake memory; resonance half-width (Q_s f0)', str(pq['num_turns_wake']), f(psbm.fr/(2*100.)/(psbm.Qs*psbm.f0), 0))
pc = loadv('psb_q100_candidates')
allr = [dict(Qp=p['Qp'], bucket=p['bucket'], g=m['g'], re=m['re'], gp=m['gp'], rep=m['rep'], dev=m['dev_pct']) for p in pc['points'] for m in p['matched']]
allr += [dict(Qp=r['Qp'], bucket=r['bucket'], g=r['g'], re=r['re'], gp=r['gp'], rep=r['rep'], dev=r['dev_pct']) for r in pc['runs']]


def at_line(r):
    pt = next(p for p in pc['points'] if p['Qp'] == r['Qp'] and p['bucket'] == r['bucket'])
    return next(m['start_dQ'][1] for m in pt['modes'] if m['start'] == 'line' and abs(m['p'][0] - r['rep']) < 1e-4 and abs(m['p'][1] - r['gp']) < 1e-4)


shifted = [r for r in allr if (r['Qp'] <= -2 or r['Qp'] == 4) and abs(r['re']) >= 1]
under = [100*(1 - at_line(r)/r['gp']) for r in shifted]              # relative to the growth rate at the mode's own frequency
under_g = [100*(1 - at_line(r)/r['g']) for r in shifted]            # relative to the tracked one
P("Q_r = 100, Q' <= -2 and +4: lines underestimate the modes shifted by |Re dQ| >= Q_s by from, to (of the own-frequency rate)",
  f(min(under), 0), pct(max(under), 0))
P('(the same relative to the tracked rates, from, to; the runs)', f(min(under_g), 0), pct(max(under_g), 0), str(len(shifted)))
other = {(r['Qp'], r['bucket'], round(r['rep'], 6)): 100*(1 - at_line(r)/r['gp']) for r in allr if r not in shifted and abs(r['re']) >= 1}
P("(the modes shifted by |Re dQ| >= Q_s at the other Q': Q', bucket, underestimate %)", *[f"{Qp:+.0f} {b} {x:.1f}" for (Qp, b, _), x in sorted(other.items())])
P('Q_r = 100: modes iterated if growing at least ... as fast as the fastest at the lines', pct(100*pc['floor'], 0))
ad = sorted(abs(r['dev']) for r in allr)
P('Q_r = 100: simulations agreeing within (all but the largest deviation; nearest); of; the largest deviation', str(len(ad) - 1), pct(ad[-2]), str(len(ad)),
  pct(ad[-1], 0))
P('(the next largest deviation, %)', f(ad[-2], 3))
P('Q_r = 100: seeded runs include every mode within ... of the fastest at its own frequency', pct(100*pc['within'], 0))
# the near-degenerate modes of the linearized bucket at Q' = -8 (stored with the runs: the seeded ones)
m8 = sorted(int(round(q['re'])) for q in runs if q['bucket'] == 'linear' and q['Qp'] == -8. and q['run'] == 'seeded')
P("Q' = -8, linearized: modes seeded (m); the rule: every mode within ... of the most unstable", ', '.join(f"{m:+d}" if m else '0' for m in m8),
  pct(100*psbm.within, 0))
P("(Q' = -8, linearized: the slowest seeded mode below the fastest by, %)",
  f(100*(1 - min(q['gp'] for q in runs if q['bucket'] == 'linear' and q['Qp'] == -8. and q['run'] == 'seeded')
         /max(q['gp'] for q in runs if q['bucket'] == 'linear' and q['Qp'] == -8.)), 2))
# psb.py seeds every mode within psb.within of the fastest and kicks where there is none: a kicked run means that the
# fastest mode grows at least that much faster than every other
for Qp in (-8., -7.):
    nlr = [q for q in runs if q['bucket'] == 'nonlinear' and q['Qp'] == Qp]
    P(f"Q' = {Qp:+.0f}, sinusoidal: the single run is kicked, on mode m; so that mode grows faster than every other by at least",
      str(len(nlr) == 1 and nlr[0]['run'] == 'kicked'), f"{round(nlr[0]['re']):.0f}", pct(100*psbm.within, 0))
pe = {Qp: 100*(v - 1) for Qp, v in eff_ps.items()}
P("bucket effect from, to (except +1, +2); at Q' = +1; at +2", f(min(v for k, v in pe.items() if k not in (1., 2.)), 0),
  pct(max(v for k, v in pe.items() if k not in (1., 2.)), 0), 'unchanged' if abs(pe[1.]) < 0.05 else pct(pe[1.]), pct(pe[2.]))
d_ps = np.array([dev(q) for q in runs if q.get('g') is not None and q.get('gp')])
seeded_ps = [q for q in runs if q['run'] == 'seeded']; kicked_ps = [q for q in runs if q['run'] != 'seeded']
P('simulations; linearized (seeded); sinusoidal (seeded)', str(len(runs)), str(sum(q['bucket'] == 'linear' for q in runs)),
  str(sum(q['bucket'] == 'linear' for q in seeded_ps)), str(sum(q['bucket'] == 'nonlinear' for q in runs)), str(sum(q['bucket'] == 'nonlinear' for q in seeded_ps)))
P('against the solvers at the modes\' frequencies: range; rms; within 0.5%', *rng(d_ps), pct(np.sqrt(np.mean(d_ps**2))), str(int(np.sum(abs(d_ps) < 0.5))))
P('real tune shifts within (Q_s): seeded; kicked', f(max(abs(q['re'] - q['rep']) for q in seeded_ps), 3), f(max(abs(q['re'] - q['rep']) for q in kicked_ps), 2))
d_line = np.array([100*(q['g']/q['gp_line'] - 1) for q in runs if q.get('g') is not None and q.get('gp_line')])
P('against the solvers at the betatron lines, up to', spct(d_line.min()))

# ----------------------------------------------------------------------------------------------------------------------
section('Sec. V F: coupled-bunch modes')
P('bunches; N_b; tau_b (ns); Q\'; Q_r', 'eight' if cbm.M == 8 else str(cbm.M), sci(cbm.Nb, 0), f(c.taub*1e9, 0), f"{cbm.Qp:+.0f}", f(cbm.Qres, 0))
P('f_r (MHz); R_s (Ohm/m); width f_r/Q_r (kHz); line spacing of one n_x (kHz)', f(cbm.fr/1e6, 3), sci(cbm.Rs, 1), f(cbm.fr/cbm.Qres/1e3, 0),
  f(cbm.M*c.f0/1e3, 0))
damped = [-e_cb[('nonlinear', nx)]['gp'] for nx in range(4)] + [-e_cb[('linear', nx)]['gp'] for nx in range(4)]
driven = [e_cb[('nonlinear', nx)]['gp'] for nx in range(4, 8)] + [e_cb[('linear', nx)]['gp'] for nx in range(4, 8)]
P('n_x = 0-3 damped at from, to; n_x = 4-7 driven at from, to (Q_s)', f(min(damped), 3), f(max(damped), 3), f(min(driven), 3), f(max(driven), 3))
cx = loadv('cb_crosstalk')
P('next l = 0 eigenvalue at most (of the shifted one)', pct(100*max(r['spectrum_near_zero']['second_over_largest'] for r in cx)))
P('largest |dQ| over the resonance width (orders of magnitude below)',
  f"{np.floor(np.log10((cbm.fr/cbm.Qres)/(max(abs(complex(q['rep'], -q['gp'])) for q in cb)*c.Qs*c.f0))):.0f}")
P('truncation; eigenvalues; reproduced at', f"({cbm.lmax},{cbm.nmax_check})", str(sum(q['held'] for q in cb)), f"({cbm.lmax},{cbm.nmax})")
P('initial content of the other n_x (of the seed), from, to', *same_exp(min(r['content_start'] for r in cx), max(r['content_start'] for r in cx)))
lk = {(r['bucket'], r['nx']): r['leak_max'] for r in cx}
P('largest other-mode content, run of n_x = 0; other runs (of the seeded amplitude)', f(max(lk[(b, 0)] for b in ('linear', 'nonlinear')), 0),
  f(max(v for (b, nx), v in lk.items() if nx != 0), 1))
P('projection separates them to; contribution to the fitted signal at most', sci(max(r['cross_into_nx'] for r in cx), 1), sci(max(r['contamination_max'] for r in cx), 1))
P('macroparticles per bunch; slices per bunch; turns of memory; e-foldings growing, damped, of the fastest', sci(cbm.num_particles, 0),
  str(cbm.num_slices), str(cbm.num_turns_wake), str(cbm.efoldings_growing), str(cbm.efoldings_damped), str(cbm.efoldings_fastest))
dcb = np.array([dev(q) for q in cb])
P('all 16: deviation from, to; real tune shifts within (Q_s)', *rng(dcb, 2), sci_up(max(abs(q['re'] - q['rep']) for q in cb), 1))
eff_cbp = {nx: 100*(v - 1) for nx, v in eff_cb.items()}
eff_cbt = {nx: 100*(e_cb[('nonlinear', nx)]['g']/e_cb[('linear', nx)]['g'] - 1) for nx in range(8)}
P('bucket effect from, to (except n_x = 4); n_x = 4; simulations reproduce each within', sf(min(v for k, v in eff_cbp.items() if k != 4)),
  spct(max(v for k, v in eff_cbp.items() if k != 4)), spct(eff_cbp[4]), pct(max(abs(eff_cbt[k] - eff_cbp[k]) for k in eff_cbp), 3))
# the l = 0 kernel at the lines nearest the resonance: xi = 2 pi (f - f_xi)/v, f_xi = Q' f0/eta, at the resonance and its image
f_xi = cbm.Qp*c.f0/c.eta
lines = [cbm.fr, -cbm.fr]
P('xi sigma_z at the resonance; at its image; f - f_xi (MHz) there (about)', *[f(abs(2*np.pi*(fl - f_xi)/c.v*c.sigma_z), 1) for fl in lines],
  *[sig((fl - f_xi)/1e6) for fl in lines])

# ======================================================================================================================
section('Sec. V G (summary) and Sec. VI (conclusion)')
P('rates in the scans (with repeats); distinct; within 1%; all but n within 4%', str(n_all), str(n_distinct), str(n_w1), str(n_distinct - n_w4))
beyond = [(q, dev(q)) for q in entries if abs(dev(q)) >= 4]
withre = [q for q in entries if q.get('rep') is not None and q.get('re') is not None]
P('(same modes: runs whose tracked Re dQ is within 0.02 Q_s of the predicted mode, of the runs with both; the resistive-wall rows store no Re dQ)',
  str(sum(abs(q['re'] - q['rep']) <= 0.02 for q in withre)), str(len(withre)))
P('(beyond 4%)', *sorted({f"{q['bucket']} Q'={q.get('Qp'):+.0f} {'damping ' + str(q['damping']) if 'damping' in q else 'slices ' + str(q.get('slices'))}: {sf(x)}%" for q, x in beyond}))
P('bucket effects: smallest factor (resistive wall); TMCI threshold; largest factor (bunch length)', f(min(eff_rw.values()), 2), pct(-th_shift, 0), f(max(eff_bl.values()), 1))
P('conclusion: s = 300 (s = 100 for the coupled-bunch mode); growth between; Re dQ between', str(S_MAX), str(S_LAW[1]), sci(g300[0], 1), sci_up(g300[-1]),
  sci(re300[0], 1), sci_up(re300[-1]))
P('conclusion: sixteen coupled-bunch rates to', pct(np.abs(dcb).max(), 2))
P('conclusion: LHC bunch fills of the bucket; TMCI threshold lowered by; head-tail factor up to', pct(100*c.taub*1e9/bucket_ns, 0), pct(-th_shift), f(max(eff_bl.values()), 2))
raised = [tb for tb, x in nl_rep.items() if x < -0.5 and eff_p[tb] > 1]       # Q' = +5: mode -1 the fastest and raised
P("conclusion: mode -1 raised in bunches of up to (ns); lowered at 1.6, 1.7 ns to (of the linear value)", f(max(raised), 1),
  f(r_m1[1.6], 2), f(r_m1[1.7], 2))
P('conclusion: PSB growth rates raised by up to', pct(max(pe.values()), 0))
P('conclusion: coupled-bunch modes changed by below', pct(np.ceil(max(abs(v) for v in eff_cbp.values())), 0))
