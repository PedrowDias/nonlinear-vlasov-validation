"""Fig. 7 and Table I: the PSB at 160 MeV (below transition), a bunch of 503 ns full length with a line density parabolic in
sin(hz/2R), N_b = 4e12, a transverse resonator of Q_r = 10, Q' = -8 to +4. The reference of the linearized bucket is the
nonlinear solver in a bucket stretched 200 times (its linear limit), with the exact parabola in z of the same full length; that
of the sinusoidal bucket the nonlinear solver; both at (13,32), held against (13,24); DELPHI at (13,40) for comparison. The
impedance is sampled at the betatron lines, Z((Q_y0 + k) omega_0), or at each mode's own complex frequency, Z(Omega + k omega_0),
iterated to self-consistency. Where several modes grow within 8% of the fastest, each is seeded with its eigenvector (that of
the matrix with the impedance at the lines); otherwise the bunch is kicked; 25 e-foldings. Writes data/psb.json.

    python3 psb.py
"""
import numpy as np
import xtrack as xt
import xwakes as xw
from scipy.special import ellipk, ellipkinc

import common as c
from common import DELPHI
import nonlinear
from particle_param import proton_param
from nonlinear import Jz_of_kappa, g0_parabolic, G_I_Cache, compute_impedance_matrix, compute_damper_matrix, \
    computes_coef, eigenmodesDELPHI, resonator_impedance, basis_arrays

circ = 157.07
Ekin = 160e6
alpha_p = 0.06016
h = 1
V_rf = 8e3
taub = 503e-9
Nb = 4e12
Qx, Qy = 4.26, 4.45
Rt, fr, Qres = 4e6, 1.72e6, 10.
nemitt = 2e-12

Qpscan = np.arange(-8., 4.1, 1.)
lmax, nmax, nmax_check = 13, 24, 32
nmax_delphi = 40
abseps = 1e-4
within = 0.08

num_particles = 100_000
num_slices = 200
num_turns_wake = 20
max_turns = 100_000
efoldings = 25
kick_sigma = 1e-3
seed_sigma = 0.3
amp_rescale = 1e-3
seed = 1

electron, _, _, mc2 = proton_param()
gamma = Ekin*electron/mc2 + 1
beta0 = np.sqrt(1 - 1/gamma**2)
E0 = gamma*xt.PROTON_MASS_EV
v = beta0*c.clight
omega0 = 2*np.pi*v/circ
f0 = omega0/(2*np.pi)
R = circ/(2*np.pi)
eta = alpha_p - 1/gamma**2
Qs = np.sqrt(abs(eta)*h*V_rf/(2*np.pi*beta0**2*E0))
omegas = Qs*omega0
Qfrac = Qx - np.floor(Qx)
z_b = v*taub/2                                  # half length of the parabolic bunch
bets = abs(eta)*circ/(2*np.pi*Qs)
zeta_half_range = np.pi*R/h
beta_x = R/Qx
sigma_x = np.sqrt(nemitt/(beta0*gamma)*beta_x)
kappa_edge = np.sin(h*z_b/(2*R))

freq = np.hstack(([1e-5], np.arange(5e2, 5e7+5e2, 5e2)))
Z = resonator_impedance(Rt, fr, Qres, freq)
cd, cz = computes_coef(f0, 0., 1., 0., 1, Nb, gamma, Qx)
line_sampling = nonlinear.Zy_interp


def sampled_at(dQ):
    """the resonator at omega + dQ Qs omega0, dQ complex, in place of the table interpolation"""
    def Z_at(omega, Z, freq):
        w = np.asarray(omega, dtype=complex) + dQ*Qs*omega0
        wr = 2*np.pi*fr
        return (Rt*wr/w)/(1 - 1j*Qres*(wr/w - w/wr))
    return Z_at


def bucket(h_b):
    ke = np.sin(h_b*z_b/(2*R))
    Jz_max = float(Jz_of_kappa(np.array([ke]), R, Qs, h_b, eta)[0])
    g0f = g0_parabolic(ke, R, Qs, h_b, eta, p=0.5)
    return dict(h=h_b, Jz_max=Jz_max, g0f=g0f, cache=G_I_Cache(nmax_check, g0f, R, Qs, h_b, eta, Jz_upper=Jz_max))


buckets = {'linear': bucket(h/200.), 'nonlinear': bucket(h)}


def eigenmodes(b, Qp, n, vectors=False):
    matZ, _, b['cache'] = compute_impedance_matrix(lmax, n, 0, 1, Qp*omega0/eta, omega0, Qfrac, b['g0f'], Z, freq, R, Qs, b['h'], eta, v,
                                                   Jz_upper=b['Jz_max'], cache=b['cache'], abseps=abseps*Qs*omega0/abs(cz))
    matd = compute_damper_matrix(lmax, n, 0, 1, Qp*omega0/eta, omega0, Qfrac, b['g0f'], R, Qs, b['h'], eta, Jz_upper=b['Jz_max'], cache=b['cache'])
    e, vec = eigenmodesDELPHI(lmax, n, matd, matZ, cd, cz, omegas, R, Qs, b['h'], eta, Jz_upper=b['Jz_max'], flageigenvect=vectors)
    return e/omega0/Qs, vec


def predict(mode, Qp):
    """the most unstable mode and those within 8% of it: [(dQ line sampling, dQ complex sampling, eigenvector)]"""
    b = buckets[mode]
    nonlinear.Zy_interp = line_sampling
    e, vec = eigenmodes(b, Qp, nmax_check, vectors=True)
    e_check, _ = eigenmodes(b, Qp, nmax)
    real = [k for k in np.argsort(e.imag) if e[k].imag < 0 and c.held(e[k], e_check)]
    group = []
    for k in real:
        if e[k].imag > (1 - 1.5*within)*e[real[0]].imag:
            break
        p = e[k]
        for _ in range(8):
            nonlinear.Zy_interp = sampled_at(p)
            ec, _ = eigenmodes(b, Qp, nmax_check)
            p, p_old = ec[np.argmin(np.abs(ec - p))], p
            if abs(p - p_old) < 1e-6:
                break
        nonlinear.Zy_interp = line_sampling
        group.append((e[k], p, vec[:, :, k]))
    fastest = min(m[1].imag for m in group)
    return [m for m in group if m[1].imag <= (1 - within)*fastest]


def delphi(Qp):
    """DELPHI at (lmax, nmax_delphi) with its Laguerre-series parabola, most unstable dQ/Qs"""
    g, a, b = DELPHI.longdistribution_decomp(taub, typelong='parabolicline')
    r = DELPHI.eigenmodesDELPHI_converged_scan([Qp], [0], [0.], [Nb], [omegas], [0.], 1, omega0, Qx, gamma, eta, a, b, taub, g, Z, freq,
                                               kmax=1, kmaxplot=1, crit=5e-2, abseps=abseps, fixed_lmax_nmax=(lmax, nmax_delphi))
    return complex(r['tuneshift_most'].squeeze())/Qs


def one_turn(mode, Qp):
    return xt.LineSegmentMap(longitudinal_mode='linear_fixed_rf' if mode == 'linear' else 'nonlinear', length=circ, qx=Qx, qy=Qy,
                             betx=beta_x, bety=R/Qy, dqx=Qp, dqy=0., momentum_compaction_factor=alpha_p, voltage_rf=[V_rf],
                             frequency_rf=[h*f0], phase_rf=[0.])


def parabolic(mode, n, rng):
    """g0 = (1 - kappa^2/kappa_edge^2)^(1/2) in each bucket, whose line density is a parabola"""
    z_max, d_max = (np.pi*R/h, 2*Qs/(h*abs(eta))) if mode == 'nonlinear' else (z_b, z_b/bets)
    zs, ds = [], []
    while sum(len(z) for z in zs) < n:
        z = rng.uniform(-z_max, z_max, 4*n)
        d = rng.uniform(-d_max, d_max, 4*n)
        if mode == 'nonlinear':
            u = 1 - (np.sin(h*z/(2*R))**2 + (h*eta*d/(2*Qs))**2)/kappa_edge**2
        else:
            u = 1 - (z/z_max)**2 - (d/d_max)**2
        keep = (u > 0) & (rng.uniform(0, 1, 4*n) < np.sqrt(np.clip(u, 0, None)))
        zs.append(z[keep]); ds.append(d[keep])
    return np.concatenate(zs)[:n], np.concatenate(ds)[:n]


def mode_pattern(mode, Qp, vec, zeta, delta):
    """x - j beta px of the mode on the particles, the seed formula of Sec. V B of the paper:
    exp(-j Q' z/(eta R)) sum_l exp(-j l phi) sum_n c_ln psi_n(J), with (J, phi) the action-angle variables of Sec. II C,
    sin(hz/2R) = kappa sn(v), h|eta| delta/(2 Qs) = kappa cn(v), v = K - 2K phi/pi, so that phi = pi/2 - (pi/2K) F(am(v))
    with am(v) = atan2(sn, cn). The solver's l-th block is the e^{-j l phi} harmonic of this phi (phi_dot = Omega_s, with
    the sign of eta), on either side of transition."""
    b = buckets[mode]
    sn = np.sin(b['h']*zeta/(2*R))
    cn = b['h']*abs(eta)*delta/(2*Qs)
    kappa = np.clip(np.hypot(sn, cn), 1e-12, 1 - 1e-12)
    phi = np.pi/2 - np.pi/(2*ellipk(kappa**2))*ellipkinc(np.arctan2(sn/kappa, cn/kappa), kappa**2)
    Jz = Jz_of_kappa(kappa, R, Qs, b['h'], eta)
    w = np.zeros(len(zeta), dtype=complex)
    for l in range(-lmax, lmax+1):
        w += np.exp(-1j*l*phi)*(vec[l+lmax] @ basis_arrays(abs(l), nmax_check, Jz, b['Jz_max'])[0])
    return w*np.exp(-1j*Qp*omega0/(eta*v)*zeta)


def run(mode, Qp, growth_predicted, vec=None):
    """kicked (vec None, fit over the last quarter) or seeded with vec (fit over the last half)"""
    wake = xw.WakeResonator(kind='dipolar_x', r=Rt, q=Qres, f_r=fr)
    wake.configure_for_tracking(zeta_range=(-zeta_half_range, zeta_half_range), num_slices=num_slices,
                                num_turns=num_turns_wake, circumference=circ)
    line = c.build({'map': one_turn(mode, Qp), 'wake': wake}, p0c=np.sqrt(E0**2 - xt.PROTON_MASS_EV**2))
    particles = c.matched(line, Nb, num_particles, z_b/np.sqrt(5.), parabolic(mode, num_particles, np.random.default_rng(seed)), nemitt=nemitt,
                          seed=seed)
    if vec is None:
        particles.x += kick_sigma*sigma_x
    else:
        w = mode_pattern(mode, Qp, vec, particles.zeta, particles.delta)      # x - j beta px
        w *= seed_sigma*sigma_x/np.max(np.abs(w))
        particles.x += np.real(w)
        particles.px += -np.imag(w)/beta_x
    num_turns = c.turns(efoldings, growth_predicted, max_turns, Qs=Qs)
    signal = c.track(line, particles, num_turns, beta_x=beta_x, amp_rescale=amp_rescale, memory=wake._wake_tracker.moments_data)
    return c.fit(signal, num_turns - (num_turns//4 if vec is None else num_turns//2), Qfrac=Qfrac, Qs=Qs)[:2]


if __name__ == '__main__':
    groups = {(mode, Qp): predict(mode, Qp) for mode in buckets for Qp in Qpscan}
    out = dict(runs=[], delphi=[])
    for Qp in Qpscan:
        d = delphi(Qp)
        out['delphi'].append(dict(Qp=Qp, re=d.real, g=-d.imag))
        for mode in buckets:
            group = groups[(mode, Qp)]
            seeded = len(group) > 1
            for p, pc, vec in group:
                g, re = run(mode, Qp, -pc.imag, vec if seeded else None)
                out['runs'].append(dict(Qp=Qp, bucket=mode, run='seeded' if seeded else 'kicked', g=g, re=re,
                                        gp=-pc.imag, rep=pc.real, gp_line=-p.imag, rep_line=p.real))
                print(f"Q' = {Qp:+.0f}, {mode}, {'seeded' if seeded else 'kicked'}: growth {g:.4f} ({-pc.imag:.4f}; lines {-p.imag:.4f})", flush=True)
                c.save('psb', out)
