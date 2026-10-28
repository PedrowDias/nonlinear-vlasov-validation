"""Fig. 8 and Table I: eight equidistant LHC bunches, N_b = 2e11, Q' = +5, a resonator of Q_r = 300 centred on the n_x = 2
line near 20 MHz. For each coupled-bunch mode n_x, the l = 0 mode the resonator shifts: the reference of the linearized bucket
is the nonlinear solver in a bucket stretched 200 times (its linear limit), that of the sinusoidal bucket the nonlinear solver;
both at (6,48), held against (6,32). Each n_x is tracked on its own, seeded with its eigenvector on a quiet-start bunch of zero
transverse emittance, and fitted with a matrix pencil. Writes data/coupled_bunch.json.

    python3 coupled_bunch.py
"""
import numpy as np
import xtrack as xt
import xwakes as xw
from scipy.special import ellipk, ellipkinc, ellipj

import common as c
from nonlinear import Jz_of_kappa, computes_coef, resonator_impedance, basis_arrays

M = 8
Nb = 2e11
Qp = 5.
Qres = 300.
nx_res = 2
Rs = 1.1e11
lmax, nmax, nmax_check = 6, 32, 48
abseps = 1e-4

num_particles = 20_000           # per bunch
num_slices = 300
num_turns_wake = 3
efoldings_growing = 6
efoldings_damped = 3
efoldings_fastest = 12
seed_amplitude = 1e-6
amp_rescale = 1e-3
pencil_components = 6

spacing = c.circ/M
xi_chi = Qp*c.omega0/(c.eta*c.v)
nxscan = np.arange(M)
fr = (int(round(2e7/(M*c.f0)))*M + nx_res + c.Qfrac)*c.f0
j = np.arange(int(5e9/c.f0))
freq = np.unique(np.hstack(((j + c.Qfrac)*c.f0, (j[1:] - c.Qfrac)*c.f0)))      # every line |(p M + nx + Qfrac) f0|
Z = resonator_impedance(Rs, fr, Qres, freq)
cd, cz = computes_coef(c.f0, 0., 1., 0., M, Nb, c.gamma, c.Qx)
buckets = {'linear': c.gaussian_bucket(nmax_check, h_b=c.h/200.), 'nonlinear': c.gaussian_bucket(nmax_check)}


def predict(mode, nx):
    """the l = 0 mode the resonator shifts: (dQ/Qs, eigenvector, whether nmax reproduces it)"""
    b = buckets[mode]
    e, vec = c.eigen(b, lmax, nmax_check, *c.matrices(b, lmax, nmax_check, Qp, abseps*c.Qs*c.omega0/abs(cz), nx, M, freq, Z), cd, cz, vectors=True)
    e_check = c.eigen(b, lmax, nmax, *c.matrices(b, lmax, nmax, Qp, abseps*c.Qs*c.omega0/abs(cz), nx, M, freq, Z), cd, cz)[0]
    near_zero = np.flatnonzero(np.abs(e.real) < 0.5)
    k = near_zero[np.argmax(np.abs(e[near_zero].imag))]
    return e[k], vec[:, :, k], c.held(e[k], e_check)


def one_turn(mode):
    """the linearized bucket: linearized rf, periodic in zeta as a multi-bunch beam needs"""
    return xt.LineSegmentMap(longitudinal_mode='linear_fixed_rf' if mode == 'linear' else 'nonlinear', length=c.circ, qx=c.Qx, qy=c.Qy,
                             betx=c.beta_x, bety=c.R/c.Qy, dqx=Qp, dqy=0., momentum_compaction_factor=c.alpha_p, voltage_rf=[c.V_rf],
                             frequency_rf=[c.h*c.f0], phase_rf=[np.pi])


def quiet_bunch(mode, n):
    """J from the inverse cumulative distribution of g0 at (i + 1/2)/n, the angle on a golden-ratio sequence, along the
    pendulum orbits; in the linearized bucket the orbits are cut inside the rf bucket, where the linearized rf wraps"""
    b = buckets[mode]
    kappa_cut = b['kappa_edge'] if mode == 'nonlinear' else min(b['kappa_edge'], np.sin(b['h']*0.99*c.zeta_half_range/(2*c.R)))
    kappa_grid = np.linspace(0, kappa_cut, 400001)
    w = b['g0f'](kappa_grid)
    F = np.concatenate(([0.], np.cumsum(0.5*(w[1:] + w[:-1])*np.diff(Jz_of_kappa(kappa_grid, c.R, c.Qs, b['h'], c.eta)))))
    i = np.arange(n)
    kappa = np.interp((i + 0.5)/n, F/F[-1], kappa_grid)
    sn, cn, _, _ = ellipj(4*ellipk(kappa**2)*((i*(np.sqrt(5) - 1)/2) % 1), kappa**2)
    return 2*c.R*np.arcsin(kappa*sn)/b['h'], 2*c.Qs*kappa*cn/(b['h']*abs(c.eta))      # z, delta from (J_z, phi), Sec. II C


def mode_pattern(mode, vec, zeta, delta):
    """x - j beta px of the mode on the particles, the seed formula of Sec. V B of the paper:
    exp(-j Q' z/(eta R)) sum_l exp(-j l phi) sum_n c_ln psi_n(J), with (J, phi) the action-angle variables of Sec. II C,
    sin(hz/2R) = kappa sn(v), h|eta| delta/(2 Qs) = kappa cn(v), v = K - 2K phi/pi, so that phi = pi/2 - (pi/2K) F(am(v))
    with am(v) = atan2(sn, cn). The solver's l-th block is the e^{-j l phi} harmonic of this phi."""
    b = buckets[mode]
    sn = np.sin(b['h']*zeta/(2*c.R))
    cn = b['h']*abs(c.eta)*delta/(2*c.Qs)
    kappa = np.clip(np.hypot(sn, cn), 1e-12, 1 - 1e-12)
    phi = np.pi/2 - np.pi/(2*ellipk(kappa**2))*ellipkinc(np.arctan2(sn/kappa, cn/kappa), kappa**2)
    Jz = Jz_of_kappa(kappa, c.R, c.Qs, b['h'], c.eta)
    w = np.zeros(len(zeta), dtype=complex)
    for l in range(-lmax, lmax+1):
        w += np.exp(-1j*l*phi)*(vec[l+lmax] @ basis_arrays(abs(l), nmax_check, Jz, b['Jz_max'])[0])
    return w*np.exp(-1j*xi_chi*zeta)


def pencil(signal):
    """matrix-pencil fit of the signal with the betatron line shifted to zero, decimated to ~1000 samples: the component
    carrying most of the signal at turn 0, (Re dQ/Qs, growth/Qs)"""
    d = max(1, len(signal)//1000)
    s = (signal*np.exp(-2j*np.pi*c.Qfrac*np.arange(len(signal))))[::d]
    L = len(s)//3
    Y = np.array([s[i:i+L+1] for i in range(len(s) - L)])
    V = np.linalg.svd(Y, full_matrices=False)[2][:pencil_components].T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    A = np.linalg.lstsq(np.vander(z, len(s), increasing=True).T, s, rcond=None)[0]
    lam = np.log(z[np.argmax(np.abs(A))])/d
    return lam.imag/(2*np.pi*c.Qs), lam.real/(2*np.pi*c.Qs)


def run(mode, nx, dQ, vec, growth_fastest):
    wake = xw.WakeResonator(kind='dipolar_x', r=Rs, q=Qres, f_r=fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=num_slices,
                                filling_scheme=np.ones(M, dtype=int), bunch_spacing_zeta=spacing,
                                num_turns=num_turns_wake, circumference=c.circ)
    line = c.build({'map': one_turn(mode), 'wake': wake}, p0c=np.sqrt(c.E0**2 - xt.PROTON_MASS_EV**2))
    z, d = quiet_bunch(mode, num_particles)
    a = mode_pattern(mode, vec, z, d)                                # x - j beta px of the mode
    a *= seed_amplitude/np.max(np.abs(a))
    phase = np.exp(2j*np.pi*(nx + c.Qfrac + dQ*c.Qs)/M)            # bunch k passes k T/M after bunch 0
    seed = np.concatenate([a*phase**k for k in range(M)])           # x - j beta px
    particles = line.build_particles(zeta=np.concatenate([z - k*spacing for k in range(M)]), delta=np.tile(d, M),
                                     x=seed.real, px=-seed.imag/c.beta_x, weight=Nb/num_particles)
    bunch = particles.particle_id//num_particles
    px0 = particles.px.copy()                  # the initial state as the previous passage, without its kick
    wake.track(particles)
    particles.px[:] = px0

    growth = -dQ.imag
    num_turns = int(min(efoldings_growing/growth if growth > 0 else efoldings_damped/-growth,
                        efoldings_fastest/growth_fastest)/(2*np.pi*c.Qs))
    projection = np.exp(-2j*np.pi*(nx + c.Qfrac)*np.arange(M)/M)/M
    memory = wake._wake_tracker.moments_data                       # indexed by moment name, not by row
    signal = np.empty(num_turns, dtype=complex)
    scale = 1.
    for turn in range(num_turns):
        centroids = (np.bincount(bunch, weights=particles.x, minlength=M)
                     - 1j*c.beta_x*np.bincount(bunch, weights=particles.px, minlength=M))/num_particles
        signal[turn] = projection @ centroids*scale
        line.track(particles, num_turns=1)
        if np.abs(centroids).max() > amp_rescale:
            particles.x *= 1e-3
            particles.px *= 1e-3
            memory['x'][:] *= 1e-3                                  # the dipolar source is num_particles * <x>
            scale *= 1e3
    return pencil(signal)


if __name__ == '__main__':
    predicted = {(mode, nx): predict(mode, nx) for mode in buckets for nx in nxscan}
    growth_fastest = max(-p[0].imag for p in predicted.values())
    rows = []
    for mode in buckets:
        for nx in nxscan:
            dQ, vec, held = predicted[(mode, nx)]
            re, g = run(mode, nx, dQ, vec, growth_fastest)
            rows.append(dict(nx=nx, bucket=mode, g=g, gp=-dQ.imag, re=re, rep=dQ.real, held=bool(held)))
            print(f"nx = {nx}, {mode}: growth {g:+.5f} ({-dQ.imag:+.5f}){'' if held else ', not held at nmax'}", flush=True)
            c.save('coupled_bunch', rows)
