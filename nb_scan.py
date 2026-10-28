"""Chromaticity paragraph of Sec. V C: the bucket effect on mode -1 against intensity, Q' = +5, N_b = 3e10 to 3e11.
Solvers: DELPHI (most unstable mode, and mode -1 from its spectrum: below about 1.2e11 the most unstable mode of both buckets is
an l = 0 radial mode), the nonlinear solver at (6,48) held against (6,32), and two constant-Omega_s variants of it that keep
the nonlinear orbits and g0 but replace Omega_s(J_z) by omega_s or by its average over g0, which split the bucket effect into
orbits and bunch shape, mean synchrotron frequency, and spread. Also the average and rms spread of Omega_s over the bunch.
Simulations: kicked runs in both buckets at every intensity (12 e-foldings, stopped at 1 mm, fitted over the last quarter but
at least the last 3000 turns), then, at 3e10 and 1e11, mode -1 seeded in the sinusoidal bucket with a pure l = -1 pattern
(the kicked run is taken over by the l = 0 radial mode there), 80 000 turns, fitted by the slope over the last quarter and
over turns 10-50%, and by a matrix pencil from 5% of the run on. Writes data/validation/nb_scan.json.

    python3 nb_scan.py solvers      # the solvers only (about an hour on an Apple M3), keeping the stored simulations
    python3 nb_scan.py              # solvers and simulations (a night)
"""
import os, sys, json
import numpy as np
import xwakes as xw
from scipy.special import ellipk, ellipkinc

import common as c
import seeded
from common import DELPHI
from nonlinear import computes_coef, Jz_of_kappa

Qp = 5.
Nbscan = np.array([3e10, 5e10, 1e11, 1.5e11, 2e11, 2.5e11, 3e11])
Nb_seeded = [3e10, 1e11]
lmax, nmax, nmax_check = 6, 32, 48
abseps = 1e-4

num_particles = 100_000
num_slices = 300
max_turns = 600_000
efoldings = 12
kick_sigma = 1e-3
seed_sigma = 0.3                # as psb.py; the pattern's centroid is then 0.019 sigma_x, 4.3x the shot-noise centroid of 1e5 particles (0.0045 sigma_x)
seeded_turns = 80_000           # the seeded runs are fitted with the matrix pencil before the faster l = 0 mode takes over
amp_max = 1e-3
fit_turns = 3000
seed = 1
OUT = os.path.join(c.DATA, 'validation')


def mode_minus1(e):
    """the eigenvalues of e with Re dQ/Qs within 0.25 of -1 (mode -1), in the order of e"""
    return [x for x in e if abs(x.real + 1) < 0.25]


def delphi_spectrum():
    """Nb -> DELPHI's eigenvalues dQ/Qs at Q' (matrices built once, as in tmci.py)"""
    g, a, bb = DELPHI.longdistribution_decomp(c.taub, typelong='Gaussian')
    omegaksi = Qp*c.omega0/c.eta
    matZ = DELPHI.compute_impedance_matrix(lmax, nmax, 0, 1, omegaksi, c.omega0, c.Qfrac, a, bb, c.taub, g, c.Z, c.freq, abseps=abseps/10)[0]
    matd = DELPHI.compute_damper_matrix(lmax, nmax, 0, 1, omegaksi, c.omega0, c.Qfrac, a, bb, c.taub, g)

    def f(Nb):
        cd, cz = DELPHI.computes_coef(c.f0, 0., bb, g[0], 1., c.taub, 0., 1, Nb, c.gamma, c.Qx)
        return DELPHI.eigenmodesDELPHI(lmax, nmax, matd, matZ, cd, cz, c.omegas)[0]/c.omega0/c.Qs
    return f


def mean_omegas_ratio(b):
    """<Omega_s>/omega_s over the bunch: Omega_s(kappa) = s_eta pi omega_s/(2K(kappa^2)) weighted by g0 dJ (and its rms spread)"""
    kap = np.linspace(1e-6, b['kappa_edge']*(1 - 1e-9), 200001)
    J = Jz_of_kappa(kap, c.R, c.Qs, c.h, c.eta)
    w = b['g0f'](kap)*np.gradient(J, kap)
    ratio = np.pi/(2*ellipk(kap**2))
    mean = np.trapezoid(w*ratio, kap)/np.trapezoid(w, kap)
    return mean, np.sqrt(np.trapezoid(w*(ratio - mean)**2, kap)/np.trapezoid(w, kap))


def predict():
    b = c.gaussian_bucket(nmax_check)
    delphi_eigs = delphi_spectrum()
    om_mean, om_rms = mean_omegas_ratio(b)
    print(f"<Omega_s>/omega_s = {om_mean:.4f}, rms spread {om_rms:.4f}", flush=True)
    cz0 = computes_coef(c.f0, 0., 1., 0., 1, 3e11, c.gamma, c.Qx)[1]
    mats = {n: c.matrices(b, lmax, n, Qp, abseps*c.Qs*c.omega0/abs(cz0)) for n in (nmax, nmax_check)}
    out = []
    for Nb in Nbscan:
        cd, cz = computes_coef(c.f0, 0., 1., 0., 1, Nb, c.gamma, c.Qx)
        e = {n: c.eigen(b, lmax, n, *mats[n], cd, cz)[0] for n in mats}
        held = c.most_unstable_held(e[nmax_check], e[nmax])
        m1 = mode_minus1(held)
        # constant-Omega_s diagnostic: same matrices (orbits and g0), l*D replaced by l*s_eta*Omega_c with Omega_c constant
        # (D is then s_eta Omega_c * identity). Two choices of the constant: omega_s (removes the spread AND the mean
        # detuning <Omega_s> - omega_s) and <Omega_s> (removes the spread only).
        matZ, matd = mats[nmax_check]
        base = (cd*matd + cz*matZ).reshape(2*lmax+1, nmax_check+1, 2*lmax+1, nmax_check+1)
        cst = {}
        for label, fac in (('omega_s', 1.), ('mean_Omega_s', om_mean)):
            Mat = base.copy()
            for l in range(-lmax, lmax+1):
                Mat[l+lmax, :, l+lmax, :] += l*np.sign(c.eta)*c.omegas*fac*np.eye(nmax_check+1)
            ec = np.linalg.eigvals(Mat.reshape(((2*lmax+1)*(nmax_check+1),)*2))/c.omega0/c.Qs
            cst[label] = sorted(mode_minus1(ec[ec.imag < 0]), key=lambda x: x.imag)
        ec_m1 = cst['omega_s']
        d = c.delphi_most([Qp], [0.], Nb, c.taub, lmax, nmax)[0, 0]
        ed = delphi_eigs(Nb)
        dm1 = sorted(mode_minus1(ed[ed.imag < 0]), key=lambda x: x.imag)
        row = dict(Nb=Nb, delphi=[d.real, -d.imag], delphi_mode_m1=[dm1[0].real, -dm1[0].imag] if dm1 else None,
                   nonlinear_most=[held[0].real, -held[0].imag] if held else None,
                   nonlinear_mode_m1=[m1[0].real, -m1[0].imag] if m1 else None,
                   constant_Omegas_mode_m1=[ec_m1[0].real, -ec_m1[0].imag] if ec_m1 else None,
                   constant_meanOmegas_mode_m1=[cst['mean_Omega_s'][0].real, -cst['mean_Omega_s'][0].imag] if cst['mean_Omega_s'] else None,
                   mean_Omegas_over_omegas=om_mean, rms_Omegas_over_omegas=om_rms)
        out.append(row)
        print(f"Nb = {Nb:.2e}: DELPHI {d.real:+.4f}/{-d.imag:.5f}, mode -1 {row['delphi_mode_m1']} | nonlinear most {row['nonlinear_most']} | "
              f"mode -1 held {row['nonlinear_mode_m1']} | constant omega_s mode -1 {row['constant_Omegas_mode_m1']} | "
              f"constant <Omega_s> mode -1 {row['constant_meanOmegas_mode_m1']}", flush=True)
    return out


def seed_l_minus1(zeta, delta):
    """x - j beta px = A exp(-j Q' z/(eta R)) exp(+j phi) sqrt(u): a pure l = -1 perturbation, lowest radial function"""
    sn = np.sin(c.h*zeta/(2*c.R))
    cn = c.h*abs(c.eta)*delta/(2*c.Qs)
    kappa = np.clip(np.hypot(sn, cn), 1e-12, 1 - 1e-12)
    phi = np.pi/2 - np.pi/(2*ellipk(kappa**2))*ellipkinc(np.arctan2(sn/kappa, cn/kappa), kappa**2)
    u = Jz_of_kappa(kappa, c.R, c.Qs, c.h, c.eta)/Jz_of_kappa(np.array([1 - 1e-12]), c.R, c.Qs, c.h, c.eta)[0]
    return np.exp(-1j*Qp*c.omega0/(c.eta*c.v)*zeta)*np.exp(+1j*phi)*np.sqrt(u)


def run(mode, Nb, growth_predicted, seed_l1=False):
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=num_slices)
    line = c.build({'map': c.one_turn(mode, Qp), 'wake': wake})
    zd = c.boltzmann(num_particles, c.alpha, np.random.default_rng(seed)) if mode == 'nonlinear' else None
    particles = c.matched(line, Nb, num_particles, c.sigma_z, zd, seed=seed)
    if seed_l1:
        w = seed_l_minus1(particles.zeta, particles.delta)
        w *= seed_sigma*c.sigma_x/np.max(np.abs(w))
        particles.x += np.real(w)
        particles.px += -np.imag(w)/c.beta_x
    else:
        particles.x += kick_sigma*c.sigma_x
    n = seeded_turns if seed_l1 else c.turns(efoldings, max(growth_predicted, 1e-3), max_turns)
    signal = c.track(line, particles, n, amp_max=np.inf if seed_l1 else amp_max, amp_rescale=amp_max if seed_l1 else np.inf)
    i1 = len(signal)
    g, re, grew = c.fit(signal, max(i1//3, i1 - max(fit_turns, i1//4)))
    # for a seeded run also the slope over the first part and the matrix pencil (from 5% in), before another mode takes over
    g_early = np.polyfit(np.arange(i1//10, i1//2), np.log(np.abs(signal[i1//10:i1//2])), 1)[0]/(2*np.pi*c.Qs) if seed_l1 else None
    pen = seeded.pencil(signal[i1//20:], re_target=-1.)[:2] if seed_l1 else None
    c.save_signal(f"nb_scan_{Nb:.1e}_{mode}_{'seeded' if seed_l1 else 'kicked'}", signal)
    return g, re, grew, g_early, i1, pen


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'nb_scan.json')
    tracking = json.load(open(path))['tracking'] if os.path.exists(path) else []
    out = dict(Qp=Qp, solvers=predict(), tracking=tracking)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    if 'solvers' in sys.argv[1:]:
        sys.exit()
    out['tracking'] = []
    for row in out['solvers']:                       # the kicked runs, both buckets, every intensity
        Nb = row['Nb']
        for mode in ('nonlinear', 'linear'):
            gp = row['delphi'][1] if mode == 'linear' else (row['nonlinear_most'] or [0, 0])[1]
            g, re, grew, _, n, _ = run(mode, Nb, gp)
            out['tracking'].append(dict(Nb=Nb, bucket=mode, run='kicked', g=g if grew else None, re=re, gp=gp, turns=n))
            print(f"Nb = {Nb:.2e}, {mode}, kicked: growth {g:.5f} ({gp:.5f}), Re {re:+.4f}", flush=True)
            with open(os.path.join(OUT, 'nb_scan.json'), 'w') as f:
                json.dump(out, f, indent=1, default=lambda o: o.item())
    for row in out['solvers']:                       # mode -1 seeded in the sinusoidal bucket at the lowest intensities
        Nb = row['Nb']
        if Nb in Nb_seeded:
            gp = (row['nonlinear_mode_m1'] or [0, 0])[1]
            g, re, grew, g_early, n, (re_p, g_p) = run('nonlinear', Nb, gp, seed_l1=True)
            out['tracking'].append(dict(Nb=Nb, bucket='nonlinear', run='seeded l=-1', g=g, g_early=g_early, re=re, gp=gp, g_pencil=g_p, re_pencil=re_p,
                                        gp_linear=(row.get('delphi_mode_m1') or row['delphi'])[1], turns=n))
            print(f"Nb = {Nb:.2e}, nonlinear, seeded l=-1: growth {g:+.5f} (early {g_early:+.5f}; pencil {re_p:+.4f}/{g_p:+.5f}; nonlinear {gp:.5f}, "
                  f"linear mode -1 {(row.get('delphi_mode_m1') or row['delphi'])[1]:.5f}), Re {re:+.4f}", flush=True)
            with open(os.path.join(OUT, 'nb_scan.json'), 'w') as f:
                json.dump(out, f, indent=1, default=lambda o: o.item())
