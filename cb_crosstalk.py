"""Sec. V F: how cleanly the eight-bunch simulations separate the coupled-bunch modes. Each n_x is run as in
coupled_bunch.py (seeded with its eigenvector on a quiet-start bunch), and every turn the bunch centroids are projected on all
eight coupled-bunch modes, P_k. Per run (both buckets, every n_x):
  cross_seed          |P_k|/|P_nx| at turn 0, largest over k != n_x: the projection's own cross-talk on the seeded pattern
                      (the eight patterns, each with its own dQ, are not orthogonal under P);
  content_start       the initial content of the other growing modes k, |beta_k|/|P_nx(0)| from a least-squares fit
                      P_k(t) = alpha P_nx(t) + beta_k z_k^t + beta_0 z_0^t (z_k = exp[2 pi j (Q_f + dQ_k Q_s)] the predicted
                      eigenvalue of n_x = k, z_0 its unshifted modes), each row divided by |P_nx(t)|; a damped mode's content
                      decays below the fit's resolution, so only the growing modes are used;
  leak_max            the largest |P_k|/|P_nx| over the run, k != n_x;
  cross_into_nx       |P_nx . pattern_k|/|P_k . pattern_k|, largest over k != n_x, and contamination_max, the largest
                      sum_k cross_into_nx(k) |P_k(t)|/|P_nx(t)|: what the other modes contribute to the fitted signal;
  spectrum_near_zero  the solver's l = 0 family of that n_x: the shifted mode and the second-largest |dQ| relative to it.
Writes data/validation/cb_crosstalk.json and the projections to data/signals/ (16 runs).

    python3 cb_crosstalk.py
"""
import os, json, time
import numpy as np
import xtrack as xt
import xwakes as xw

import common as c
import coupled_bunch as cb

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'cb_crosstalk.json')
CASES = [(mode, nx) for mode in ('linear', 'nonlinear') for nx in range(cb.M)]
P = np.exp(-2j*np.pi*np.outer(np.arange(cb.M) + c.Qfrac, np.arange(cb.M))/cb.M)/cb.M       # row k: projection on mode k


def predict_with_spectrum(mode, nx):
    """coupled_bunch.predict (same matrices, same selection), also returning a summary of the l = 0 family (eigenvectors with
    more than half their norm in l = 0, unperturbed at dQ = 0): the paper's "in each n_x a single l = 0 mode is shifted" is
    checked by the second-largest |dQ| among them relative to the largest"""
    b = cb.buckets[mode]
    e, vec = c.eigen(b, cb.lmax, cb.nmax_check, *c.matrices(b, cb.lmax, cb.nmax_check, cb.Qp, cb.abseps*c.Qs*c.omega0/abs(cb.cz), nx, cb.M, cb.freq, cb.Z), cb.cd, cb.cz, vectors=True)
    e_check = c.eigen(b, cb.lmax, cb.nmax, *c.matrices(b, cb.lmax, cb.nmax, cb.Qp, cb.abseps*c.Qs*c.omega0/abs(cb.cz), nx, cb.M, cb.freq, cb.Z), cb.cd, cb.cz)[0]
    near_zero = np.flatnonzero(np.abs(e.real) < 0.5)
    k = near_zero[np.argmax(np.abs(e[near_zero].imag))]
    l0 = lambda i: float(np.sum(np.abs(vec[cb.lmax, :, i])**2)/np.sum(np.abs(vec[:, :, i])**2))
    fam = [i for i in range(len(e)) if l0(i) > 0.5]                 # the l = 0 family: unperturbed at dQ = 0 in both buckets
    shifts = sorted(fam, key=lambda i: -abs(e[i]))
    spec = dict(n_l0=len(fam), largest=[e[shifts[0]].real, -e[shifts[0]].imag], largest_l0_share=l0(shifts[0]),
                second=[e[shifts[1]].real, -e[shifts[1]].imag] if len(shifts) > 1 else None,
                second_over_largest=float(abs(e[shifts[1]])/abs(e[shifts[0]])) if len(shifts) > 1 else 0., selected_is_largest=bool(k == shifts[0]),
                selected_l0_share=l0(k))
    return e[k], vec[:, :, k], c.held(e[k], e_check), spec


def pattern(nx, dQ):
    return np.exp(2j*np.pi*(nx + c.Qfrac + dQ*c.Qs)*np.arange(cb.M)/cb.M)


def run_all_projections(mode, nx, dQ, vec, growth_fastest):
    """coupled_bunch.run, with the projections on every coupled-bunch mode kept"""
    wake = xw.WakeResonator(kind='dipolar_x', r=cb.Rs, q=cb.Qres, f_r=cb.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=cb.num_slices,
                                filling_scheme=np.ones(cb.M, dtype=int), bunch_spacing_zeta=cb.spacing,
                                num_turns=cb.num_turns_wake, circumference=c.circ)
    line = c.build({'map': cb.one_turn(mode), 'wake': wake}, p0c=np.sqrt(c.E0**2 - xt.PROTON_MASS_EV**2))
    z, d = cb.quiet_bunch(mode, cb.num_particles)
    a = cb.mode_pattern(mode, vec, z, d)
    a *= cb.seed_amplitude/np.max(np.abs(a))
    phase = np.exp(2j*np.pi*(nx + c.Qfrac + dQ*c.Qs)/cb.M)
    seed = np.concatenate([a*phase**k for k in range(cb.M)])
    particles = line.build_particles(zeta=np.concatenate([z - k*cb.spacing for k in range(cb.M)]), delta=np.tile(d, cb.M),
                                     x=seed.real, px=-seed.imag/c.beta_x, weight=cb.Nb/cb.num_particles)
    bunch = particles.particle_id//cb.num_particles
    px0 = particles.px.copy()
    wake.track(particles)
    particles.px[:] = px0
    growth = -dQ.imag
    num_turns = int(min(cb.efoldings_growing/growth if growth > 0 else cb.efoldings_damped/-growth,
                        cb.efoldings_fastest/growth_fastest)/(2*np.pi*c.Qs))
    memory = wake._wake_tracker.moments_data
    signals = np.empty((num_turns, cb.M), dtype=complex)
    scale = 1.
    for turn in range(num_turns):
        centroids = (np.bincount(bunch, weights=particles.x, minlength=cb.M)
                     - 1j*c.beta_x*np.bincount(bunch, weights=particles.px, minlength=cb.M))/cb.num_particles
        signals[turn] = (P @ centroids)*scale
        line.track(particles, num_turns=1)
        if np.abs(centroids).max() > cb.amp_rescale:
            particles.x *= 1e-3; particles.px *= 1e-3; memory['x'][:] *= 1e-3; scale *= 1e3
    return signals


def analyse(S, mode, nx, dQs):
    """the quantities of the docstring from the stored projections S (turns x M) and the predicted dQ/Qs of every n_x"""
    t = np.arange(len(S))
    own = S[:, nx]
    norm = np.abs(own)
    others = [k for k in range(cb.M) if k != nx]
    ratio = np.abs(S[:, others])/norm[:, None]
    cross_into = {k: float(np.abs(P[nx] @ pattern(k, dQs[k]))/np.abs(P[k] @ pattern(k, dQs[k]))) for k in others}
    content, resid = {}, {}
    for k in others:
        zk = np.exp(2j*np.pi*(c.Qfrac + dQs[k]*c.Qs))
        z0 = np.exp(2j*np.pi*c.Qfrac)
        A = np.stack([own, zk**t, z0**t], axis=1)/norm[:, None]
        y = S[:, k]/norm
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        content[k] = float(np.abs(coef[1])/norm[0])
        resid[k] = float(np.linalg.norm(A @ coef - y)/np.linalg.norm(y))
    contamination = np.max(sum(cross_into[k]*np.abs(S[:, k]) for k in others)/norm)
    i0 = int(np.argmax(ratio[0])); imax = np.unravel_index(np.argmax(ratio), ratio.shape)[1]
    growing = [k for k in others if dQs[k].imag < 0]       # the content that matters is that of the modes that grow; a damped
    kc = max(growing, key=content.get)                    # mode's content decays below the fit's resolution (residual 0.2-1.0)
    return dict(bucket=mode, nx=nx, turns=len(S), dQ=[dQs[nx].real, -dQs[nx].imag], damped=bool(dQs[nx].imag > 0),
                cross_seed=float(ratio[0].max()), cross_seed_mode=others[i0],
                content_start=content[kc], content_start_mode=kc, content_fit_residual_max=max(resid[k] for k in growing),
                content_start_all={str(k): content[k] for k in others}, content_fit_residual_all={str(k): resid[k] for k in others},
                leak_max=float(ratio.max()), leak_max_mode=others[imax], leak_median_of_turn_max=float(np.median(ratio.max(axis=1))),
                cross_into_nx=max(cross_into.values()), contamination_max=float(contamination))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    full = {(mode, nx): predict_with_spectrum(mode, nx) for mode in cb.buckets for nx in cb.nxscan}
    predicted = {key: v[:3] for key, v in full.items()}
    stored = {(q['bucket'], q['nx']): complex(q['rep'], -q['gp']) for q in c.load('coupled_bunch')}
    worst = max(abs(predicted[k][0] - stored[k]) for k in stored)
    print(f"predictions against data/coupled_bunch.json: largest difference {worst:.1e} Qs" + ('' if worst < 1e-9 else ' (NOT the stored values)'), flush=True)
    growth_fastest = max(-dQ.imag for dQ, _, _ in predicted.values())
    rows = []
    for mode, nx in CASES:
        t0 = time.time()
        dQ, vec, _ = predicted[(mode, nx)]
        S = run_all_projections(mode, nx, dQ, vec, growth_fastest)
        os.makedirs(c.SIGNALS, exist_ok=True)
        np.save(os.path.join(c.SIGNALS, f'cb_crosstalk_{mode}_nx{nx}.npy'), S)
        r = analyse(S, mode, nx, {k: predicted[(mode, k)][0] for k in cb.nxscan})
        r['spectrum_near_zero'] = full[(mode, nx)][3]
        rows.append(r)
        json.dump(rows, open(PATH, 'w'), indent=1, default=lambda o: o.item())
        print(f"{mode:>9} nx = {nx} ({'damped' if r['damped'] else 'growing'}): {r['turns']} turns; projection cross-talk at the start "
              f"{r['cross_seed']:.1e}; initial content of the growing other modes (fit) {r['content_start']:.1e} (k = {r['content_start_mode']}, residual "
              f"{r['content_fit_residual_max']:.1e}); largest other-mode content over the run {r['leak_max']:.2g} of the seeded (k = {r['leak_max_mode']}); "
              f"cross-talk into nx {r['cross_into_nx']:.1e}, contamination of the fitted mode <= {r['contamination_max']:.1e} [{(time.time()-t0)/60:.1f} min]", flush=True)
    dm = [r for r in rows if r['damped']]; gr = [r for r in rows if not r['damped']]
    print(f"all {len(rows)} runs: projection cross-talk {min(r['cross_seed'] for r in rows):.1e}-{max(r['cross_seed'] for r in rows):.1e}; "
          f"initial content {min(r['content_start'] for r in rows):.1e}-{max(r['content_start'] for r in rows):.1e}; "
          f"cross-talk into nx <= {max(r['cross_into_nx'] for r in rows):.1e}; contamination <= {max(r['contamination_max'] for r in rows):.1e}")
    fmt = lambda rs: ', '.join('%d: %.2g' % (r['nx'], r['leak_max']) for r in rs)
    print(f"largest other-mode content, per seeded nx: damped {fmt(dm)}; growing {fmt(gr)}")
    print(f"l = 0 family: the second-largest |dQ| is at most {max(r['spectrum_near_zero']['second_over_largest'] for r in rows):.1e} of the "
          f"shifted mode's; the selected mode is the most shifted one in {sum(r['spectrum_near_zero']['selected_is_largest'] for r in rows)} of {len(rows)} cases")
    print('done')
