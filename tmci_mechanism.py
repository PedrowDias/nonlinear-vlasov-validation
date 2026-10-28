"""Mode-coupling paragraph of Sec. V C: the -6.1% shift of the TMCI threshold (i) split, as the factor on mode -1 in nb_scan.py,
by the constant-Omega_s variants of the nonlinear solver: Omega_s(J_z) replaced by omega_s (the orbits and the bunch shape
only), by its average over g0 (adds the mean synchrotron frequency), and in full (adds the spread about that mean); (ii) under
the other two matchings of matching.py (same rms emittance, same rms momentum spread). Thresholds as in tmci.py: the
intensity at which the most unstable growth rate reaches 0.02 Q_s, (lmax, nmax) = (4,24), Q' = 0. Solver only, one to two
hours on an Apple M3. Reads data/tmci.json and data/validation/matching.json; writes data/validation/tmci_mechanism.json.

    python3 tmci_mechanism.py
"""
import os, json, time
import numpy as np
from scipy.optimize import brentq

import common as c
import tmci
import nb_scan
from nonlinear import computes_coef, g0_gaussian, Jz_upper_from_bunch, G_I_Cache

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'tmci_mechanism.json')
lmax, nmax, Qp = tmci.lmax, tmci.nmax, tmci.Qp


def most_unstable_growth(b, matZ, matd, Nb, constant=None, om_mean=None):
    """-Im dQ/Qs of the most unstable eigenvalue; constant: None (full Omega_s(J)), 'omega_s' or 'mean_Omega_s' (D -> constant)"""
    cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
    if constant is None:
        e = c.eigen(b, lmax, nmax, matZ, matd, cd, cz)[0]
    else:
        fac = 1. if constant == 'omega_s' else om_mean
        Mat = (cd*matd + cz*matZ).reshape(2*lmax+1, nmax+1, 2*lmax+1, nmax+1).copy()
        for l in range(-lmax, lmax+1):
            Mat[l+lmax, :, l+lmax, :] += l*np.sign(c.eta)*c.omegas*fac*np.eye(nmax+1)
        e = np.linalg.eigvals(Mat.reshape(((2*lmax+1)*(nmax+1),)*2))/c.omega0/c.Qs
    return -e.imag.min()                          # the most unstable eigenvalue has the most negative Im


def threshold(b, matZ, matd, constant=None, om_mean=None):
    return brentq(lambda Nb: most_unstable_growth(b, matZ, matd, Nb, constant, om_mean) - tmci.threshold_Qs, 3e11, 4e11, xtol=1e6)


def bucket_from_alpha(alpha, sigma_z):
    Jz_max, kappa_edge = Jz_upper_from_bunch(sigma_z, c.R, c.Qs, c.h, c.eta)
    g0f = g0_gaussian(alpha, c.R, c.Qs, c.h, c.eta, kappa_max=kappa_edge)
    return dict(h=c.h, alpha=alpha, kappa_edge=kappa_edge, Jz_max=Jz_max, g0f=g0f, cache=G_I_Cache(nmax, g0f, c.R, c.Qs, c.h, c.eta, Jz_upper=Jz_max))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    stored = c.load('tmci')['threshold']                     # {'linear': DELPHI, 'nonlinear': solver}, in 1e11
    out = dict(stored_thresholds_1e11=stored, variants=[])
    cz_max = computes_coef(c.f0, 0., 1., 0., 1, tmci.Nbscan.max(), c.gamma, c.Qx)[1]
    # (i) mechanism, same sigma_z as the paper
    b = c.gaussian_bucket(nmax)
    om_mean = nb_scan.mean_omegas_ratio(b)[0]
    matZ, matd = c.matrices(b, lmax, nmax, Qp, tmci.abseps*c.Qs*c.omega0/abs(cz_max))
    for label, constant in (('full (paper)', None), ('Omega_s -> omega_s (orbits and shape only)', 'omega_s'), ('Omega_s -> <Omega_s> (adds the mean detuning)', 'mean_Omega_s')):
        t0 = time.time()
        N0 = threshold(b, matZ, matd, constant, om_mean)
        out['variants'].append(dict(matching='same sigma_z (paper)', variant=label, threshold_1e11=N0/1e11, shift_pct=100*(N0/1e11/stored['linear'] - 1)))
        print(f"{label:>52}: threshold {N0/1e11:.4f}e11, {100*(N0/1e11/stored['linear'] - 1):+.2f}% from DELPHI's {stored['linear']:.4f}e11 [{(time.time()-t0)/60:.0f} min]", flush=True)
        json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    v = {x['variant']: x['shift_pct'] for x in out['variants']}
    shape, mean, spread = v['Omega_s -> omega_s (orbits and shape only)'], v['Omega_s -> <Omega_s> (adds the mean detuning)'], v['full (paper)']
    out['decomposition_points'] = dict(orbits_and_shape=shape, mean_detuning=mean - shape, spread=spread - mean)
    print(f"decomposition of the {spread:+.2f}% shift: orbits and shape {shape:+.2f}, mean detuning {mean-shape:+.2f}, spread {spread-mean:+.2f} points", flush=True)
    # (ii) matching conventions (alphas from matching.json)
    m = json.load(open(os.path.join(OUT, 'matching.json')))
    for row in m['matchings']:
        if row['matching'].startswith('same sigma_z'):
            continue
        t0 = time.time()
        bb = bucket_from_alpha(row['alpha'], row['sigma_z_cm']/100)
        mZ, md = c.matrices(bb, lmax, nmax, Qp, tmci.abseps*c.Qs*c.omega0/abs(cz_max))
        N0 = threshold(bb, mZ, md)
        out['variants'].append(dict(matching=row['matching'], variant='full', threshold_1e11=N0/1e11, shift_pct=100*(N0/1e11/stored['linear'] - 1), sigma_z_cm=row['sigma_z_cm']))
        print(f"{row['matching']:>52}: threshold {N0/1e11:.4f}e11, {100*(N0/1e11/stored['linear'] - 1):+.2f}% from DELPHI (sigma_z {row['sigma_z_cm']:.2f} cm) [{(time.time()-t0)/60:.0f} min]", flush=True)
        json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    print('done')
