"""Figs. 1-8 of the paper from data/*.json, written to figures/."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)


def load(name):
    with open(os.path.join(HERE, 'data', name + '.json')) as f:
        return json.load(f)


plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3.6, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'legend.frameon': False, 'legend.handlelength': 2.2,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'axes.grid': False,
    'pdf.fonttype': 42})
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8984'
W1, W2 = 3.375, 7.0                     # one and two columns, inches
LIN = dict(color=BLUE, ls='--')
NL = dict(color=ORANGE, ls='-')
TLIN = dict(marker='o', ls='none', mfc='white', mec=BLUE, mew=0.9, color=BLUE)
TNL = dict(marker='s', ls='none', mfc=ORANGE, mec=ORANGE, color=ORANGE, ms=3.2)
GROWTH = r'$-\mathrm{Im}\,\Delta Q/Q_s$'
RE = r'$\mathrm{Re}\,\Delta Q/Q_s$'
D = dict(tmci=load('tmci')['rows'], chroma=load('chromaticity'), bunchlength=load('bunch_length'), damper=load('damper'),
         cb=load('coupled_bunch'))


def rows(key, **sel):
    return [r for r in D[key] if all(r[k] == v for k, v in sel.items())]


def tag(ax, s, loc, y=None):
    x, y0 = {'ul': (0.03, 0.95), 'ur': (0.97, 0.95), 'll': (0.03, 0.05), 'lr': (0.97, 0.05)}[loc]
    y = y0 if y is None else y
    ax.text(x, y, s, transform=ax.transAxes, va='top' if y > 0.5 else 'bottom', ha='left' if x < 0.5 else 'right', fontsize=8)


def solver_and_tracking(ax, key, x, y, ypred, skip=(), split=None, growing_only=False, **sel):
    """solver lines and tracking markers for both buckets"""
    for bucket, ls, ts, lab_s, lab_t in (('linear', LIN, TLIN, 'DELPHI (linear)', 'tracking, linearized rf'),
                                         ('nonlinear', NL, TNL, 'nonlinear solver', 'tracking, sinusoidal rf')):
        r = [q for q in rows(key, bucket=bucket, **sel) if q[x] not in skip]
        xs = np.array([q[x] for q in r]); yp = np.array([np.nan if q[ypred] is None else q[ypred] for q in r])
        o = np.argsort(xs); xs, yp = xs[o], yp[o]
        if split is not None:
            k = np.searchsorted(xs, split); xs, yp = np.insert(xs, k, np.nan), np.insert(yp, k, np.nan)
        ax.plot(xs, yp, label=lab_s, **ls)
        keep = [q for q in r if q[y] is not None and not (growing_only and q['g'] is None)]
        kicked = [q for q in keep if q.get('run') != 'seeded mode -1']
        seeded = [q for q in keep if q.get('run') == 'seeded mode -1']  # bunch_length: the three linearized points seeded with mode -1, drawn as diamonds
        ax.plot([q[x] for q in kicked], [q[y] for q in kicked], label=lab_t, **ts)
        if seeded:                                                    # drawn last and on top: at 1.6-1.8 ns the two buckets' points nearly coincide
            ax.plot([q[x] for q in seeded], [q[y] for q in seeded], label=lab_t + ', seeded', zorder=5, **dict(ts, marker='D', ms=4.2))


def linear_limit():
    LL = load('linear_limit')
    names = {'BB+5': "head-tail, $Q'=+5$", 'BB-5': "head-tail, $Q'=-5$", 'TMCI': r'mode coupling, $N_b=3.8\times10^{11}$',
             'RW+5': "resistive wall, $Q'=+5$", 'CB6': r'8 bunches, $n_x=6$'}
    markers = {'BB+5': 'o', 'BB-5': 's', 'TMCI': 'D', 'RW+5': '^', 'CB6': 'v'}
    colors = {'BB+5': BLUE, 'BB-5': ORANGE, 'TMCI': '#1baf7a', 'RW+5': '#6250d6', 'CB6': '#e34948'}
    fig, ax = plt.subplots(figsize=(W1, 3.3))
    for key in names:
        d_g = LL[key]['delphi'][1]
        r = np.array(LL[key]['rows'])
        ax.loglog(r[:, 1]**2, np.abs(r[:, 3] - d_g)/abs(d_g), marker=markers[key], color=colors[key], ms=3.5, lw=0.8,
                  mfc='white', mew=0.9, label=names[key])
    k2 = np.logspace(-6.5, -0.9, 10)
    ax.loglog(k2, k2, color=GREY, lw=0.7, ls='--', label=r'$\kappa_\sigma^2$')
    ax.set_xlabel(r'$\kappa_\sigma^2=\sin^2(h\sigma_z/2Rs)$'); ax.set_ylabel('growth rate: |solver / DELPHI $-$ 1|')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=2, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_linear_limit.pdf'))


def tmci():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W1, 3.15), sharex=True)
    solver_and_tracking(ax, 'tmci', 'Nb', 'g', 'gp')
    solver_and_tracking(bx, 'tmci', 'Nb', 're', 'rep')
    ax.set_ylabel(GROWTH); bx.set_ylabel(RE); bx.set_xlabel(r'$N_b$ ($10^{11}$ p)')
    ax.legend(loc='upper left')
    tag(ax, '(a)', 'lr'); tag(bx, '(b)', 'll')
    ax.set_xlim(0, 4.5)
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(OUT, 'fig_tmci.pdf'))


def chromaticity():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W1, 3.15), sharex=True)
    solver_and_tracking(ax, 'chroma', 'Qp', 'g', 'gp')
    solver_and_tracking(bx, 'chroma', 'Qp', 're', 'rep', skip=(0.,), split=0., growing_only=True)
    ax.set_ylabel(GROWTH); bx.set_ylabel(RE); bx.set_xlabel(r"$Q'$")
    ax.legend(loc='upper right')
    bx.set_xticks(np.arange(-10, 11, 5))
    tag(ax, '(a)', 'lr'); tag(bx, '(b)', 'ur')
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(OUT, 'fig_chroma.pdf'))


def bunch_length():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W1, 3.15), sharex=True)
    solver_and_tracking(ax, 'bunchlength', 'taub', 'g', 'gp', Qp=5.)
    solver_and_tracking(bx, 'bunchlength', 'taub', 'g', 'gp', Qp=-5.)
    ax.set_ylabel(GROWTH); bx.set_ylabel(GROWTH); bx.set_xlabel(r'$\tau_b$ (ns)')
    ax.set_ylim(top=1.75*ax.get_ylim()[1] - 0.75*ax.get_ylim()[0])      # room for the five legend entries above the curves and the 1.0 ns points
    ax.legend(loc='upper right')
    tag(ax, "(a) $Q'=+5$", 'll'); tag(bx, "(b) $Q'=-5$", 'll')
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(OUT, 'fig_bunchlength.pdf'))


def damper():
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W1, 3.15), sharex=True)
    for a_, Qp in ((ax, 5.), (bx, -5.)):
        solver_and_tracking(a_, 'damper', 'damping', 'g', 'gp', Qp=Qp)
        stable = [q['damping'] for q in rows('damper', bucket='nonlinear', Qp=Qp) if q['g'] is None]
        if stable:
            a_.plot(stable, np.zeros(len(stable)), marker='s', ls='none', mfc='white', mec=ORANGE, mew=0.9, ms=3.2,
                    label='sinusoidal rf: no growth')
        a_.set_ylim(bottom=-0.006)
    ax.set_ylabel(GROWTH); bx.set_ylabel(GROWTH)
    bx.set_xlabel(r'damper: rigid-mode damping rate $d/(2\pi Q_s)$')
    bx.legend(loc='upper right', handles=ax.get_legend_handles_labels()[0] + bx.get_legend_handles_labels()[0][-1:])
    tag(ax, "(a) $Q'=+5$", 'ur'); tag(bx, "(b) $Q'=-5$", 'll')
    fig.tight_layout(h_pad=0.3)
    fig.savefig(os.path.join(OUT, 'fig_damper.pdf'))


def resistive_wall():
    RW = load('resistive_wall')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.5))
    s = RW['solvers']
    ax.plot(s['Qp'], np.array(s['linear'])[:, 1], label='DELPHI (linear)', **LIN)
    ax.plot(s['Qp'], np.array(s['nonlinear'])[:, 1], label='nonlinear solver', **NL)
    nmax = max(r['slices'] for r in RW['tracking'])
    for r in RW['tracking']:
        if r['slices'] == nmax:
            ax.plot(r['Qp'], r['g'], **(TLIN if r['bucket'] == 'linear' else TNL))
    n = f'{nmax:,}'.replace(',', r'\,')
    ax.plot([], [], label=f'tracking, ${n}$ slices, linearized rf', **TLIN)
    ax.plot([], [], label=f'tracking, ${n}$ slices, sinusoidal rf', **TNL)
    ax.set_xlabel(r"$Q'$"); ax.set_ylabel(GROWTH); ax.legend(loc='upper right'); tag(ax, '(a)', 'll')
    for Qp in (5., 15.):                    # measured points only, joined by lines
        for bucket, st in (('linear', TLIN), ('nonlinear', TNL)):
            r = sorted((q for q in RW['tracking'] if q['Qp'] == Qp and q['bucket'] == bucket), key=lambda q: q['slices'])
            five = Qp == 5.
            bx.plot([q['slices'] for q in r], [q['g']/q['gp'] for q in r], color=st['color'], lw=0.8, ls='-' if five else ':',
                    label=f"$Q'=+{Qp:.0f}$, " + ('linearized' if bucket == 'linear' else 'sinusoidal'),
                    **{k: v for k, v in dict(st, mfc=st['mfc'] if five else 'white', marker=st['marker'] if five else 'D').items()
                       if k not in ('ls', 'color')})
    bx.axhline(1, color=GREY, lw=0.6)
    bx.set_xscale('log'); bx.set_xlabel(r'slices $N_{\rm sl}$'); bx.set_ylabel('tracked / solver')
    bx.legend(loc='upper right'); tag(bx, '(b)', 'll', y=0.10)            # clear of the line at 1
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_rw.pdf'))


def psb():
    P = load('psb')['runs']
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.7))
    for bucket, ls, ts in (('linear', LIN, TLIN), ('nonlinear', NL, TNL)):
        r = [q for q in P if q['bucket'] == bucket]
        Qps = sorted(set(q['Qp'] for q in r))
        lab = 'linear limit' if bucket == 'linear' else 'nonlinear solver'
        ax.semilogy(Qps, [max(q['gp'] for q in r if q['Qp'] == x) for x in Qps], label=lab, **ls)
        ax.semilogy(Qps, [max(q['gp_line'] for q in r if q['Qp'] == x) for x in Qps], color=ls['color'], ls=(0, (1, 1.5)), lw=0.9)
        kicked = [q for q in r if q['run'] == 'kicked']; seeded = [q for q in r if q['run'] == 'seeded']
        name = 'linearized' if bucket == 'linear' else 'sinusoidal'
        ax.plot([q['Qp'] for q in kicked], [q['g'] for q in kicked], label=f"tracking, {name} rf", **ts)
        ax.plot([q['Qp'] for q in seeded], [q['g'] for q in seeded], **dict(ts, mfc='white', marker='^', ms=3.4))
        shift = -0.12 if bucket == 'linear' else 0.12
        bx.plot([q['Qp'] + shift for q in r], [100*(q['g']/q['gp'] - 1) for q in r], **ts, label=f"{name} rf, " + r"$Z(\Omega+k\omega_0)$")
        bx.plot([q['Qp'] + shift for q in r], [100*(q['g']/q['gp_line'] - 1) for q in r], marker='x', ls='none', color=ts['color'],
                ms=3.0, mew=0.7, alpha=0.6, label=f"{name} rf, " + r"$Z[(Q_{y0}+k)\omega_0]$")
    ax.set_xlabel(r"$Q'$"); ax.set_ylabel(GROWTH); tag(ax, '(a)', 'lr')
    ax.legend(loc='upper left')
    bx.axhline(0, color=GREY, lw=0.6)
    bx.set_xlabel(r"$Q'$"); bx.set_ylabel('tracked / solver $-$ 1 (%)'); tag(bx, '(b)', 'ur')
    bx.set_ylim(-9, 2); bx.legend(loc='lower left')
    for a_ in (ax, bx):
        a_.set_xticks(np.arange(-8, 5, 2))
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_psb.pdf'))


def coupled_bunch():
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(W2, 2.5))
    bar = dict(marker='_', ls='none', ms=9, mew=1.4)
    for bucket, ls, ts, sh in (('linear', LIN, TLIN, -0.12), ('nonlinear', NL, TNL, 0.12)):
        r = sorted(rows('cb', bucket=bucket), key=lambda q: q['nx'])
        nx = np.array([q['nx'] for q in r])
        name = 'linearized' if bucket == 'linear' else 'sinusoidal'
        ax.plot(nx + sh, [q['gp'] for q in r], color=ls['color'], label=r'linear limit ($s=200$)' if bucket == 'linear' else 'nonlinear solver', **bar)
        ax.plot(nx + sh, [q['g'] for q in r], label=f'tracking, {name} rf', **ts)
        bx.plot(nx + sh, [q['rep'] for q in r], color=ls['color'], **bar)
        bx.plot(nx + sh, [q['re'] for q in r], **ts)
        cx.plot(nx + sh, [100*(q['g']/q['gp'] - 1) for q in r], **ts, label=f'{name} rf vs. its solver')
    lin, nl = (sorted(rows('cb', bucket=b), key=lambda q: q['nx']) for b in ('linear', 'nonlinear'))
    nx = np.arange(8)
    cx.plot(nx, [100*(b['gp']/a['gp'] - 1) for a, b in zip(lin, nl)], color='k', label='solvers: sin./lin. $-$ 1', **bar)
    cx.plot(nx, [100*(b['g']/a['g'] - 1) for a, b in zip(lin, nl)], marker='x', ls='none', ms=3.5, mew=0.8, color='k',
            label='tracking: sin./lin. $-$ 1')
    ax.axhline(0, color=GREY, lw=0.6)
    for a_ in (ax, bx, cx):
        a_.set_xlabel(r'coupled-bunch mode $n_x$'); a_.set_xticks(nx)
    ax.set_ylabel(GROWTH + ' (negative: damped)'); bx.set_ylabel(RE); cx.set_ylabel('rate: relative difference (%)')
    cx.legend(loc='center left', bbox_to_anchor=(0.0, 0.6), fontsize=7)
    tag(ax, '(a)', 'lr', y=0.10); tag(bx, '(b)', 'ul'); tag(cx, '(c)', 'ur')
    cx.set_ylim(-0.6, 2.4)
    fig.tight_layout(w_pad=0.8)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=4, columnspacing=1.5)
    fig.savefig(os.path.join(OUT, 'fig_cb.pdf'))


if __name__ == '__main__':
    for f in (linear_limit, tmci, chromaticity, bunch_length, damper, resistive_wall, psb, coupled_bunch):
        f()
