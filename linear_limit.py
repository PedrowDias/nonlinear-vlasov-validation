"""Fig. 1 and Sec. V A: the nonlinear solver in a bucket stretched s times (h -> h/s at fixed Q_s, eta and sigma_z) against
DELPHI, s = 1 to 300, at the same truncation, for five configurations: the LHC broad-band resonator at Q' = +5 and -5
(BB+5, BB-5), the merged mode just above the TMCI threshold (TMCI, N_b = 3.8e11), the resistive wall at Q' = +5 (RW+5) and
the coupled-bunch mode n_x = 6 of the eight bunches (CB6). Writes data/linear_limit.json: per configuration DELPHI's
(Re dQ, growth) and rows (s, kappa_sigma, Re dQ, growth, |dQ - dQ_DELPHI|/|dQ_DELPHI|).

    python3 linear_limit.py [BB+5 BB-5 TMCI RW+5 CB6]
"""
import sys
import numpy as np

import common as c
from common import DELPHI
from nonlinear import alpha_from_sigma_z, Jz_upper_from_bunch, g0_gaussian, G_I_Cache, compute_impedance_matrix, \
    compute_damper_matrix, computes_coef, eigenmodesDELPHI, resonator_impedance

stretch = [1, 3, 10, 30, 100, 300]
abseps = 1e-4


def config(name):
    lhc = dict(gamma=c.gamma, omega0=c.omega0, Qx=c.Qx, eta=c.eta, omegas=c.omegas, h=c.h, taub=c.taub, M=1, nx=0,
               freq=c.freq, Z=c.Z, lmax=6, nmax=32, Nb=3e11)
    if name in ('BB+5', 'BB-5'):
        return dict(lhc, Qp=float(name[2:]))
    if name == 'TMCI':
        return dict(lhc, Qp=0., Nb=3.8e11)
    if name == 'RW+5':
        import resistive_wall as rw
        return dict(gamma=rw.gamma, omega0=rw.omega0, Qx=rw.Qx, eta=rw.eta, omegas=rw.omegas, h=rw.h, taub=rw.taub, M=1, nx=0,
                    freq=rw.freq, Z=rw.Z, lmax=6, nmax=24, Nb=rw.Nb, Qp=5.)
    if name == 'CB6':                      # the resonator of coupled_bunch.py, nx = 6, the mode it shifts; lines up to 2.5 GHz
        M = 8
        fr = (int(round(2e7/(M*c.f0)))*M + 2 + c.Qfrac)*c.f0
        j = np.arange(int(2.5e9/c.f0))
        freq = np.unique(np.hstack(((j + c.Qfrac)*c.f0, (j[1:] - c.Qfrac)*c.f0)))
        return dict(lhc, M=M, nx=6, Nb=2e11, Qp=5., freq=freq, Z=resonator_impedance(1.1e11, fr, 300., freq), driven=True)


def pick(e, k):
    if k.get('driven'):
        near_zero = np.flatnonzero(np.abs(e.real) < 0.5)
        return e[near_zero[np.argmax(np.abs(e[near_zero].imag))]]
    return e[np.argmin(e.imag)]


def scan(k):
    omega0, eta, Qx, lmax, nmax, nx, M = k['omega0'], k['eta'], k['Qx'], k['lmax'], k['nmax'], k['nx'], k['M']
    Qs, f0, Qfrac = k['omegas']/omega0, omega0/(2*np.pi), Qx - np.floor(Qx)
    v = np.sqrt(1 - 1/k['gamma']**2)*c.clight
    R, sigma_z, om = v/omega0, k['taub']*v/4, k['Qp']*omega0/eta

    g, a, b = DELPHI.longdistribution_decomp(k['taub'], typelong='Gaussian')
    cd, cz = DELPHI.computes_coef(f0, 0., b, g[0], 1., k['taub'], 0., M, k['Nb'], k['gamma'], Qx)
    matZ = DELPHI.compute_impedance_matrix(lmax, nmax, nx, M, om, omega0, Qfrac, a, b, k['taub'], g, k['Z'], k['freq'], abseps=abseps)[0]
    matd = DELPHI.compute_damper_matrix(lmax, nmax, nx, M, om, omega0, Qfrac, a, b, k['taub'], g, abseps=abseps)
    d = pick(DELPHI.eigenmodesDELPHI(lmax, nmax, matd, matZ, cd, cz, k['omegas'])[0]/omega0/Qs, k)

    cd, cz = computes_coef(f0, 0., 1., 0., M, k['Nb'], k['gamma'], Qx)
    rows = []
    for s in stretch:
        h_b = k['h']/s
        Jz_max, kappa_edge = Jz_upper_from_bunch(sigma_z, R, Qs, h_b, eta)
        g0f = g0_gaussian(alpha_from_sigma_z(sigma_z, R, h_b), R, Qs, h_b, eta, kappa_max=kappa_edge)
        cache = G_I_Cache(nmax, g0f, R, Qs, h_b, eta, Jz_upper=Jz_max)
        mZ, _, cache = compute_impedance_matrix(lmax, nmax, nx, M, om, omega0, Qfrac, g0f, k['Z'], k['freq'], R, Qs, h_b, eta, v,
                                                Jz_upper=Jz_max, cache=cache, abseps=abseps*Qs*omega0/abs(cz))
        md = compute_damper_matrix(lmax, nmax, nx, M, om, omega0, Qfrac, g0f, R, Qs, h_b, eta, Jz_upper=Jz_max, cache=cache)
        e = pick(eigenmodesDELPHI(lmax, nmax, md, mZ, cd, cz, k['omegas'], R, Qs, h_b, eta, Jz_upper=Jz_max)[0]/omega0/Qs, k)
        rows.append([s, np.sin(h_b*sigma_z/(2*R)), e.real, -e.imag, abs(e - d)/abs(d)])
    return dict(delphi=[d.real, -d.imag], rows=rows)


if __name__ == '__main__':
    try:
        out = c.load('linear_limit')
    except FileNotFoundError:
        out = {}
    names = ['BB+5', 'BB-5', 'TMCI', 'RW+5', 'CB6']
    if any(a not in names for a in sys.argv[1:]):
        sys.exit(f"usage: python3 linear_limit.py [{' '.join(names)}]")
    for name in sys.argv[1:] or names:
        out[name] = scan(config(name))
        print(f"{name}: |solver/DELPHI - 1| = " + ", ".join(f"{r[4]:.1e} (s = {r[0]})" for r in out[name]['rows']), flush=True)
        c.save('linear_limit', out)
