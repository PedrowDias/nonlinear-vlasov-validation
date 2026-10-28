"""Sec. V C, the bunch in the sinusoidal bucket: its shape at the rms length of the 1-ns Gaussian (peak line density, FWHM,
rms momentum spread, excess kurtosis) and how the bucket effect at Q' = +-5 depends on the matching. Three Boltzmann bunches
g0 = exp(-alpha kappa^2): the same rms length as the Gaussian (the paper's choice), the same rms momentum spread, the same rms
longitudinal emittance; for each, the nonlinear solver's most unstable mode at (6,48), held against (6,32), against DELPHI.
Solver only. Writes data/validation/matching.json (tmci_mechanism.py reads it).

    python3 matching.py
"""
import os, json
import numpy as np
from scipy.optimize import brentq

import common as c
from nonlinear import g0_gaussian, Jz_upper_from_bunch, G_I_Cache, computes_coef

lmax, nmax, nmax_check = 6, 32, 48
abseps = 1e-4
OUT = os.path.join(c.DATA, 'validation')

# the linear-bucket Gaussian: sigma_z = 7.49 cm, sigma_delta = sigma_z/bets, emittance sigma_z sigma_delta
sz_ref = c.sigma_z
sd_ref = c.sigma_z/abs(c.bets)
eps_ref = sz_ref*sd_ref

# Boltzmann moments by quadrature on a (z, delta) grid inside the bucket
z = np.linspace(-np.pi*c.R/c.h, np.pi*c.R/c.h, 1201)
d = np.linspace(-2*c.Qs/(c.h*abs(c.eta)), 2*c.Qs/(c.h*abs(c.eta)), 1201)
Z, D = np.meshgrid(z, d, indexing='ij')
K2 = np.sin(c.h*Z/(2*c.R))**2 + (c.h*c.eta*D/(2*c.Qs))**2
inside = K2 < 1


def moments(alpha):
    w = np.where(inside, np.exp(-alpha*K2), 0.)
    w /= w.sum()
    sz = np.sqrt((w*Z**2).sum()); sd = np.sqrt((w*D**2).sum())
    lam = w.sum(axis=1); lam /= np.trapezoid(lam, z)
    kurt = np.trapezoid(lam*z**4, z)/sz**4 - 3
    # FWHM from the linearly interpolated half-maximum crossings (the grid alone is 0.1-0.6% low)
    hm = lam.max()/2; i = np.flatnonzero(lam >= hm); i0, i1 = i.min(), i.max()
    assert 0 < i0 and i1 < len(z) - 1, 'half maximum not bracketed by the grid'
    zl = z[i0-1] + (hm - lam[i0-1])*(z[i0] - z[i0-1])/(lam[i0] - lam[i0-1])
    zr = z[i1] + (lam[i1] - hm)*(z[i1+1] - z[i1])/(lam[i1] - lam[i1+1])
    fwhm = zr - zl
    return dict(sigma_z_cm=100*sz, sigma_delta=sd, emittance=sz*sd, peak_over_gaussian=lam.max()*np.sqrt(2*np.pi)*sz,
                fwhm_over_gaussian=fwhm/(2.3548*sz), excess_kurtosis=kurt)


def alpha_for(key, target):
    return brentq(lambda a: moments(a)[key] - target, 0.3, 200.)


matchings = {'same sigma_z (paper)': ('sigma_z_cm', 100*sz_ref), 'same sigma_delta': ('sigma_delta', sd_ref), 'same emittance': ('emittance', eps_ref)}
os.makedirs(OUT, exist_ok=True)
out = dict(reference=dict(sigma_z_cm=100*sz_ref, sigma_delta=sd_ref, emittance=eps_ref), delphi={}, matchings=[])
for Qp in (5., -5.):
    dq = c.delphi_most([Qp], [0.], 3e11, c.taub, lmax, nmax)[0, 0]
    out['delphi'][str(Qp)] = [dq.real, -dq.imag]
    print(f"DELPHI Q'={Qp:+.0f}: {dq.real:+.4f}/{-dq.imag:.5f}", flush=True)
cd, cz = computes_coef(c.f0, 0., 1., 0., 1, 3e11, c.gamma, c.Qx)
for name, (key, target) in matchings.items():
    alpha = alpha_for(key, target)
    m = moments(alpha)
    Jz_max, kappa_edge = Jz_upper_from_bunch(m['sigma_z_cm']/100, c.R, c.Qs, c.h, c.eta)
    g0f = g0_gaussian(alpha, c.R, c.Qs, c.h, c.eta, kappa_max=kappa_edge)
    b = dict(h=c.h, alpha=alpha, kappa_edge=kappa_edge, Jz_max=Jz_max, g0f=g0f, cache=G_I_Cache(nmax_check, g0f, c.R, c.Qs, c.h, c.eta, Jz_upper=Jz_max))
    row = dict(matching=name, alpha=alpha, **m, growth={})
    for Qp in (5., -5.):
        e = {n: c.eigen(b, lmax, n, *c.matrices(b, lmax, n, Qp, abseps*c.Qs*c.omega0/abs(cz)), cd, cz)[0] for n in (nmax, nmax_check)}
        held = c.most_unstable_held(e[nmax_check], e[nmax])
        row['growth'][str(Qp)] = [held[0].real, -held[0].imag] if held else None
        ratio = -held[0].imag/out['delphi'][str(Qp)][1] if held else float('nan')
        print(f"{name}: alpha={alpha:.3f}, sigma_z={m['sigma_z_cm']:.2f} cm, sigma_delta/ref={m['sigma_delta']/sd_ref:.3f}, "
              f"emittance/ref={m['emittance']/eps_ref:.3f}, peak x{m['peak_over_gaussian']:.3f}, FWHM x{m['fwhm_over_gaussian']:.3f}, "
              f"kurtosis {m['excess_kurtosis']:+.2f} | Q'={Qp:+.0f}: growth {row['growth'][str(Qp)]}, bucket effect x{ratio:.3f}", flush=True)
    out['matchings'].append(row)
    json.dump(out, open(os.path.join(OUT, 'matching.json'), 'w'), indent=1, default=lambda o: o.item())
print('done')
