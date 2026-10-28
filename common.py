"""Shared settings and helpers: the DELPHI path; the LHC case of the paper (6.5 TeV, broad-band resonator, 1-ns Gaussian bunch,
the parameters of the LHC broad-band example of DELPHI's own tests); solvers, tracking and output. The simulations track the
horizontal plane (Qx, dqx, dipolar_x wake), which stands for the vertical plane of the paper's formalism (Q_y0, Q'_y)."""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
SIGNALS = os.path.join(DATA, 'signals')
sys.path.insert(0, os.environ.get('DELPHI_PYTHON', os.path.join(HERE, '..', 'DELPHI', 'DELPHI_Python')))

import xtrack as xt
import xpart as xp
import DELPHI
from nonlinear import (alpha_from_sigma_z, Jz_upper_from_bunch, g0_gaussian, G_I_Cache, compute_impedance_matrix,
                       compute_damper_matrix, eigenmodesDELPHI, resonator_impedance)

clight = 299792458.

# LHC at 6.5 TeV, broad-band resonator, single Gaussian bunch of 1 ns (4 sigma)
gamma = 6927.62806135325
omega0 = 70657.55656577148
Qx, Qy = 62.3099997613612, 60.32
eta = 0.0003479538899641484
omegas = 134.87527592319398
h = 35640
taub = 1e-9
Rs, fr, Qres = 25e6, 0.8e9, 1.
nemitt = 2.5e-6

beta0 = np.sqrt(1 - 1/gamma**2)
v = beta0*clight
E0 = gamma*xt.PROTON_MASS_EV
circ = 2*np.pi*v/omega0
f0 = omega0/(2*np.pi)
R = circ/(2*np.pi)
Qs = omegas/omega0
alpha_p = eta + 1/gamma**2
sigma_z = taub*v/4
beta_x = R/Qx
sigma_x = np.sqrt(nemitt/(beta0*gamma)*beta_x)
V_rf = Qs**2*2*np.pi*beta0**2*E0/(eta*h)
bets = eta*circ/(2*np.pi*Qs)
zeta_half_range = np.pi*R/h
Qfrac = Qx - np.floor(Qx)
alpha = alpha_from_sigma_z(sigma_z, R, h)          # g0 = exp(-alpha kappa^2) in the sinusoidal bucket

# Both solvers interpolate Z linearly in f on this table: 1000 points per decade (and 1e5 between 0.1 and 5 GHz) keep the
# interpolation error of the Q = 1 resonator below 1e-5 everywhere, so that the solvers see the impedance whose exact wake
# the simulations apply (10 points per decade would bias the growth rates by up to 0.2%).
freq = np.unique(np.hstack((np.logspace(1, 13, 12001), np.linspace(1e8, 5e9, 49001))))
Z = resonator_impedance(Rs, fr, Qres, freq)


# --- solvers ------------------------------------------------------------------------------------
def delphi_most(Qpscan, dmax, Nb, taub, lmax, nmax, abseps=1e-4):
    """DELPHI, most unstable dQ/Qs per (Q', damper rate), Gaussian bunch"""
    g, a, b = DELPHI.longdistribution_decomp(taub, typelong='Gaussian')
    r = DELPHI.eigenmodesDELPHI_converged_scan(Qpscan, np.array([0]), dmax, [Nb], [omegas], [0.], 1, omega0, Qx, gamma, eta, a, b, taub, g,
                                               Z, freq, kmax=1, kmaxplot=1, crit=5e-2, abseps=abseps, fixed_lmax_nmax=(lmax, nmax))
    return r['tuneshift_most'][:, :, 0, 0, 0, 0]/Qs


def gaussian_bucket(nmax, sigma_z=sigma_z, h_b=h):
    """g0 = exp(-alpha kappa^2) of rms length sigma_z in the bucket of harmonic h_b, and its G, I cache"""
    alpha = alpha_from_sigma_z(sigma_z, R, h_b)
    Jz_max, kappa_edge = Jz_upper_from_bunch(sigma_z, R, Qs, h_b, eta)
    g0f = g0_gaussian(alpha, R, Qs, h_b, eta, kappa_max=kappa_edge)
    return dict(h=h_b, alpha=alpha, kappa_edge=kappa_edge, Jz_max=Jz_max, g0f=g0f,
                cache=G_I_Cache(nmax, g0f, R, Qs, h_b, eta, Jz_upper=Jz_max))


def matrices(b, lmax, nmax, Qp, abseps, nx=0, M=1, freq=freq, Z=Z):
    """impedance and damper matrices of the nonlinear solver in bucket b"""
    om = Qp*omega0/eta
    matZ, _, b['cache'] = compute_impedance_matrix(lmax, nmax, nx, M, om, omega0, Qfrac, b['g0f'], Z, freq, R, Qs, b['h'], eta, v,
                                                   Jz_upper=b['Jz_max'], cache=b['cache'], abseps=abseps)
    matd = compute_damper_matrix(lmax, nmax, nx, M, om, omega0, Qfrac, b['g0f'], R, Qs, b['h'], eta, Jz_upper=b['Jz_max'], cache=b['cache'])
    return matZ, matd


def eigen(b, lmax, nmax, matZ, matd, cd, cz, vectors=False):
    """eigenvalues dQ/Qs (and eigenvectors) of the nonlinear solver in bucket b"""
    e, vec = eigenmodesDELPHI(lmax, nmax, matd, matZ, cd, cz, omegas, R, Qs, b['h'], eta, Jz_upper=b['Jz_max'], flageigenvect=vectors)
    return e/omega0/Qs, vec


HELD_RE, HELD_G = 3e-3, 0.05           # the two-size criterion of Sec. V: Re dQ within 0.003 Qs, growth within 5%


def held(e, eigs_other):
    """an eigenvalue that the other basis size reproduces: within HELD_RE (Qs) in Re and HELD_G (relative) in growth"""
    o = eigs_other[np.argmin(np.abs(eigs_other - e))]
    return abs(o.real - e.real) < HELD_RE and abs(o.imag - e.imag) < HELD_G*abs(e.imag)


def most_unstable_held(e, e_check):
    """the most unstable eigenvalues of e that e_check reproduces, fastest first"""
    return [x for x in sorted((x for x in e if x.imag < 0), key=lambda x: x.imag) if held(x, e_check)]


# --- tracking -----------------------------------------------------------------------------------
def one_turn(mode, Qp):
    """one-turn map: the linearized bucket, or the sinusoidal one above transition"""
    common = dict(length=circ, qx=Qx, qy=Qy, betx=beta_x, bety=R/Qy, dqx=Qp, dqy=0.)
    if mode == 'linear':
        return xt.LineSegmentMap(longitudinal_mode='linear_fixed_qs', qs=Qs, bets=bets, **common)
    return xt.LineSegmentMap(longitudinal_mode='nonlinear', momentum_compaction_factor=alpha_p,
                             voltage_rf=[V_rf], frequency_rf=[h*f0], phase_rf=[np.pi], **common)


def build(elements, p0c=np.sqrt(E0**2 - xt.PROTON_MASS_EV**2)):
    line = xt.Line(elements=list(elements.values()), element_names=list(elements))
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=p0c)
    line.build_tracker()
    return line


def matched(line, Nb, n, sigma_z, zeta_delta=None, nemitt=nemitt, seed=1):
    """xpart's matched Gaussian bunch, with (zeta, delta) replaced if given. xpart draws from numpy's global RNG, so it is
    seeded here, so that a run is reproducible (a different seed scatters the growth rate by up to about 1%)"""
    if seed is not None:
        np.random.seed(seed)
    p = xp.generate_matched_gaussian_bunch(line=line, num_particles=n, total_intensity_particles=Nb,
                                           nemitt_x=nemitt, nemitt_y=nemitt, sigma_z=sigma_z)
    if zeta_delta is not None:
        p.zeta[:] = zeta_delta[0]
        p.update_delta(zeta_delta[1])
    return p


def boltzmann(n, alpha, rng, R=R, h=h, Qs=Qs, eta=eta):
    """(zeta, delta) from g0 = exp(-alpha kappa^2) in the sinusoidal bucket, by rejection"""
    z_max, d_max = np.pi*R/h, 2*Qs/(h*abs(eta))
    zs, ds = [], []
    while sum(len(z) for z in zs) < n:
        z = rng.uniform(-z_max, z_max, 4*n)
        d = rng.uniform(-d_max, d_max, 4*n)
        kappa2 = np.sin(h*z/(2*R))**2 + (h*eta*d/(2*Qs))**2
        keep = (kappa2 < 1) & (rng.uniform(0, 1, 4*n) < np.exp(-alpha*kappa2))
        zs.append(z[keep]); ds.append(d[keep])
    return np.concatenate(zs)[:n], np.concatenate(ds)[:n]


def turns(efoldings, growth, max_turns, floor=1e-3, Qs=Qs):
    return int(min(max_turns, efoldings/(2*np.pi*Qs*max(growth, floor))))


RESCALE = 1e3          # the factor by which track() scales the transverse coordinates down past amp_rescale


def track(line, particles, num_turns, beta_x=beta_x, amp_max=np.inf, amp_rescale=np.inf, memory=None):
    """centroid x - j beta px every turn. Stops past amp_max; past amp_rescale scales the transverse coordinates
    (and the wake's memory of them) down by 1e3, the motion being linear. `memory` is the wake tracker's moments_data
    (a CompressedProfile), whose rows are addressed by moment name: their order follows a Python set and changes from
    process to process, so memory.data[0] is 'x' in one run and 'num_particles' in the next"""
    signal = np.empty(num_turns, dtype=complex)
    scale = 1.
    for turn in range(num_turns):
        centroid = np.mean(particles.x) - 1j*beta_x*np.mean(particles.px)
        signal[turn] = centroid*scale
        line.track(particles, num_turns=1)
        if abs(centroid) > amp_max:
            return signal[:turn+1]
        if abs(centroid) > amp_rescale:
            particles.x *= 1/RESCALE
            particles.px *= 1/RESCALE
            if memory is not None:
                memory['x'][:] *= 1/RESCALE             # the dipolar source is num_particles * <x>; 'result' is recomputed
            scale *= RESCALE
    return signal


def tune_shift(signal, slope, Qfrac=Qfrac, Qs=Qs):
    """Re dQ/Qs of the dominant mode: peak of the spectrum of x - j beta px, growth removed, Hann window, 64x padding"""
    s = signal*np.exp(-slope*np.arange(len(signal)))
    n = 64*len(s)
    spectrum = np.abs(np.fft.fft(s*np.hanning(len(s)), n))
    nu = np.fft.fftfreq(n)[np.argmax(spectrum)]
    return ((nu - Qfrac)/Qs + 0.5/Qs) % (1/Qs) - 0.5/Qs


def fit(signal, i0, Qfrac=Qfrac, Qs=Qs):
    """growth rate and Re dQ in Qs from signal[i0:], and whether it grew by at least one e-folding there"""
    slope = np.polyfit(np.arange(i0, len(signal)), np.log(np.abs(signal[i0:])), 1)[0]
    return slope/(2*np.pi*Qs), tune_shift(signal[i0:], slope, Qfrac, Qs), slope*(len(signal) - i0) > 1


# --- output -------------------------------------------------------------------------------------
def row(p, g, re, grew, **keys):
    """one run: tracked growth (None if it did not grow) and Re dQ, and the prediction p = dQ/Qs (Re None if stable)"""
    assert np.isfinite(p), f'prediction {p} is not finite'
    return dict(keys, g=g if grew else None, gp=(-p.imag if -p.imag > 0 else 0.), re=re, rep=p.real if -p.imag > 0 else None)


def save(name, data):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, name + '.json'), 'w') as f:
        json.dump(data, f, indent=1, default=lambda o: o.item())


def load(name):
    with open(os.path.join(DATA, name + '.json')) as f:
        return json.load(f)


def save_signal(name, signal):
    """the centroid signal x - j beta px of a run, turn by turn (complex64), to data/signals/<name>.npy (not in the repository)"""
    os.makedirs(SIGNALS, exist_ok=True)
    np.save(os.path.join(SIGNALS, name + '.npy'), np.asarray(signal).astype(np.complex64))
