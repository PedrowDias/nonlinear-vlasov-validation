"""Chromaticity paragraph of Sec. V C: the linearized bucket against DELPHI at Q' = +8, where the scan's deviation is largest
(+1.2%). Kicked runs with chromaticity.py's draw: (1) at the scan's resolution, 1e5 macroparticles and 300 slices, 12
e-foldings; (2) four times the particles and twice the slices; (3) 25 e-foldings with the coordinate rescaling; and (4) mode -1
seeded with the eigenvector of the nonlinear solver in the bucket stretched 200 times, for 10 e-foldings of DELPHI's growth
rate, fitted by the slope over the last half and by a matrix pencil. DELPHI's four fastest modes at this Q' are stored with
them. Writes data/validation/linear_control.json (finished parts are kept when the script is restarted).

    python3 linear_control.py      # a few hours on an Apple M3
"""
import os, json, time
import numpy as np
import xwakes as xw

import common as c
import chromaticity as ch
import bunch_length as bl
import bunch_length_seeded as bs
import seeded

Qp = 8.
OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'linear_control.json')
amp_max, fit_turns = 1e-3, 3000        # kicked runs: stopped once the centroid reaches 1 mm, fitted over the last quarter but at least 3000 turns


def kicked(particles, slices, efoldings, growth_predicted, rescale=False):
    """a kicked run with chromaticity.py's draw and kick, at the given resolution and length"""
    wake = xw.WakeResonator(kind='dipolar_x', r=c.Rs, q=c.Qres, f_r=c.fr)
    wake.configure_for_tracking(zeta_range=(-c.zeta_half_range, c.zeta_half_range), num_slices=slices)
    line = c.build({'map': c.one_turn('linear', Qp), 'wake': wake})
    p = c.matched(line, ch.Nb, particles, c.sigma_z, seed=ch.seed)
    p.x += ch.kick_sigma*c.sigma_x
    n = c.turns(efoldings, growth_predicted, 200_000)
    if rescale:
        signal = c.track(line, p, n, amp_rescale=1e-3)
        i1 = len(signal)
        g, re, grew = c.fit(signal, i1 - i1//4)                  # the last quarter
    else:
        signal = c.track(line, p, n, amp_max=amp_max)
        i1 = len(signal)
        g, re, grew = c.fit(signal, max(i1//3, i1 - max(fit_turns, i1//4)))
    c.save_signal(f'linear_control_kicked_{particles}_{slices}_{efoldings}ef', signal)
    re_p, g_p, _ = seeded.pencil(signal[len(signal)//2:])         # the pencil on the last half, to see what else is in the signal
    return g, re, grew, g_p, re_p, i1


def seeded_run(growth_predicted):
    """mode -1 seeded with the eigenvector of the nonlinear solver in the bucket stretched 200 times, for 10 e-foldings of
    DELPHI's growth rate"""
    b = c.gaussian_bucket(bl.nmax_check, h_b=c.h/200.)
    ev, vec, held = bl.solver_modes(b, Qp)['mode -1']
    line, p = ch.machine('linear', Qp)
    w = seeded.mode_pattern(b, vec, p.zeta, p.delta, bl.lmax, bl.nmax_check, Qp)
    w *= seeded.SEED_SIGMA*c.sigma_x/np.max(np.abs(w))
    p.x += np.real(w)
    p.px += -np.imag(w)/c.beta_x
    signal = c.track(line, p, c.turns(seeded.EFOLDINGS, growth_predicted, 150_000), amp_rescale=seeded.AMP_RESCALE)
    re_p, g_p, in_window = seeded.pencil(signal[len(signal)//10:], re_target=ev.real)
    g_fit, re_fit, _ = c.fit(signal, len(signal)//2)
    c.save_signal('linear_control_seeded_mode-1', signal)
    return dict(solver=[ev.real, -ev.imag], held=bool(held), g_pencil=g_p, re_pencil=re_p, pencil_in_window=bool(in_window),
                g_fit=g_fit, re_fit=re_fit, turns=len(signal))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    q = [r for r in c.load('chromaticity') if r['bucket'] == 'linear' and r['Qp'] == Qp][0]
    gp, rep = q['gp'], q['rep']
    if 'delphi' not in out:
        dm = bs.delphi_modes(c.taub, Qp)
        fast = sorted((x for x in dm if x.imag < 0), key=lambda x: x.imag)[:4]
        out['delphi'] = dict(Qp=Qp, stored=dict(gp=gp, rep=rep), fastest=[[x.real, -x.imag] for x in fast])
        print("DELPHI's fastest modes: " + ', '.join(f"{x.real:+.4f}/{-x.imag:.5f}" for x in fast), flush=True)
        json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    for label, particles, slices, ef, rescale in (('repro', 100_000, 300, 12, False), ('resolution', 400_000, 600, 12, False), ('long', 100_000, 300, 25, True)):
        if label in out:
            continue
        t0 = time.time()
        g, re, grew, g_p, re_p, n = kicked(particles, slices, ef, gp, rescale)
        out[label] = dict(particles=particles, slices=slices, efoldings=ef, turns=n, g=g if grew else None, re=re, dev=100*(g/gp - 1) if grew else None,
                          g_pencil=g_p, re_pencil=re_p)
        print(f"{label:>10}: {particles} particles, {slices} slices, {ef} e-foldings, {n} turns: growth {g:.5f} ({100*(g/gp-1):+.2f}% of DELPHI's {gp:.5f}), "
              f"Re {re:+.4f} ({rep:+.4f}); pencil on the last half {re_p:+.4f}/{g_p:.5f} [{(time.time()-t0)/60:.0f} min]", flush=True)
        json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    if 'seeded' not in out:
        t0 = time.time()
        s = seeded_run(gp)
        s['dev_fit'] = 100*(s['g_fit']/gp - 1); s['dev_pencil'] = 100*(s['g_pencil']/gp - 1)
        out['seeded'] = s
        print(f"    seeded: stretched solver {s['solver'][0]:+.4f}/{s['solver'][1]:.5f}{'' if s['held'] else ' (not held)'}; slope fit {s['re_fit']:+.4f}/{s['g_fit']:.5f} "
              f"({s['dev_fit']:+.2f}% of DELPHI), pencil {s['re_pencil']:+.4f}/{s['g_pencil']:.5f} ({s['dev_pencil']:+.2f}%), {s['turns']} turns [{(time.time()-t0)/60:.0f} min]", flush=True)
        json.dump(out, open(PATH, 'w'), indent=1, default=lambda o: o.item())
    print('done')
