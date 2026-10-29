# Scripts and data for "Generalizing direct Vlasov solvers for nonlinear longitudinal dynamics"

P. de Carvalho Dias Werneck, T. Pieloni, N. Mounet

This repository holds the scripts that compute every number, table and figure of the numerical validation and the
conclusion of the paper (Secs. V and VI), and the results they write. They are a refactored version of the scripts that
produced the stored results, and reproduce them (Reproducibility, below). The solvers are DELPHI, the direct Vlasov solver for
linear synchrotron motion, and its extension to the nonlinear motion of a sinusoidal rf bucket, the subject of the paper;
the macroparticle simulations use Xsuite.

## Quick start

With the stored results in `data/`, two scripts reproduce the paper's figures and numbers in a few minutes:

```
python3 figures.py          # Figs. 1-8 -> figures/fig_*.pdf (needs numpy and matplotlib only)
python3 paper_numbers.py    # the numbers of the paper (needs the full environment below, DELPHI included)
```

`paper_numbers.py` computes, section by section, every number that the paper quotes in Secs. V and VI, Tables I and II,
the figure captions and the abstract (and the few of Secs. III, IV and Appendix C that come from a computation), from
`data/` and, for the machine and simulation parameters, from the scripts that use them, with the definition the paper
gives. It prints each next to a short description; lines in parentheses give supporting values that the paper does not
print.

## Requirements

- Python 3.13 and the packages of `requirements.txt` at the versions used for the paper:
  `pip install -r requirements.txt -c requirements-lock.txt` (`requirements-lock.txt` pins every package they pull in). The
  `xsuite` meta-package is needed: without it, xtrack 0.112 cannot build a tracker.
- DELPHI with the nonlinear solver: the `nonlinear-equations` branch of DELPHI (https://gitlab.cern.ch/IRIS/DELPHI) at commit
  `7b8184a`, the whole checkout. Its Python module loads the compiled library `DELPHI/libDELPHI.so`, which is built with
  `make lib` in the `DELPHI/DELPHI` folder and needs g++, gfortran, GSL, LAPACK and BLAS. The scripts import `DELPHI`,
  `nonlinear` and `particle_param` from the checkout's `DELPHI_Python` folder, which they look for next to this repository
  (`../DELPHI/DELPHI_Python`) or at the path in the environment variable `DELPHI_PYTHON`.
- DELPHI's Python module reads the user name from the environment variable `USER` when it is imported. Where `USER` is not
  set (in a container, for instance), set it first: `export USER=$(id -un)`.
- Memory: DELPHI at (l_max, n_max) = (6,32) or larger on the LHC broad-band impedance table needs more than 4 GB per process.
- For the coupled-bunch and PSB cases, DELPHI's nonlinear module prints `warning: compute_impedance_matrix: impedance table
  exhausted ... before convergence`. This is expected: the stored values were computed with these tables, and extending
  them (the PSB's from 50 to 200 MHz, the coupled-bunch one from 5 to 10 GHz) changes the growth rates by at most 1.2e-9
  relative and the real tune shifts by at most 8e-10 Q_s.

## Scripts

`common.py` holds the LHC case (6.5 TeV, broad-band resonator, 1-ns Gaussian bunch) and the shared solver, tracking and
output helpers; `seeded.py` how a single-bunch simulation is started (seeded with the eigenvector of the predicted mode, as
described in Sec. V B of the paper, or kicked) and fitted. The paper writes the transverse plane as the vertical one (Q_y0,
Q'_y); the scripts compute and track the horizontal plane with the same tune, chromaticity and impedance (`Qx`, the
`dipolar_x` wakes), which is the same problem. Every script states in its docstring what it computes, for which part of
the paper, and how to run it, and the validation scripts describe the fields of the file they write.

The scans of Table I and the figures:

| Paper | Script | Results | Simulations |
|---|---|---|---|
| Fig. 1, Sec. V A | `linear_limit.py` | `data/linear_limit.json` | solvers only |
| Fig. 2 | `tmci.py` | `data/tmci.json` | 62, about 4 h |
| Fig. 3 | `chromaticity.py` | `data/chromaticity.json` | 42, about 6 h |
| Fig. 4 | `bunch_length.py` | `data/bunch_length.json` | 39, about 10 h |
| Table II | `distributions.py` | `data/distributions.json` | 16, about 3 h |
| Fig. 5 | `damper.py` | `data/damper.json` | 52, about 6 h |
| Fig. 6 | `resistive_wall.py` | `data/resistive_wall.json` | 24, days |
| Fig. 7 | `psb.py` | `data/psb.json` | 69 |
| Fig. 8 | `coupled_bunch.py` | `data/coupled_bunch.json` | 16 |

The times are those of an Apple M3, one process per script. Each scan script first runs its solvers, then its simulations;
`tmci.py`, `chromaticity.py`, `bunch_length.py`, `distributions.py`, `damper.py`, `resistive_wall.py` and `nb_scan.py`
accept the argument `solvers`, which recomputes the solver predictions only and keeps the stored simulations.

The checks that the text of Sec. V quotes, each writing to `data/validation/`:

| Paper | Script | Results |
|---|---|---|
| Sec. V B: run-to-run scatter | `multiseed.py` | `multiseed.json` |
| Sec. V C: bunch shape and matching | `matching.py` | `matching.json` |
| Sec. V C, mode coupling: threshold onset, five seeds | `tmci_seeds.py` | `tmci_seeds.json` |
| Sec. V C, mode coupling: shift decomposed and matching | `tmci_mechanism.py` | `tmci_mechanism.json` |
| Sec. V C, chromaticity: Q' = +8 control | `linear_control.py` | `linear_control.json` |
| Sec. V C, chromaticity: intensity scan, constant-Omega_s solvers | `nb_scan.py` | `nb_scan.json` |
| Sec. V C, chromaticity: the l = 0 radial mode at 1e11 | `nb_l0_seeded.py` | `nb_l0_seeded.json` |
| Secs. V and V C: predictions at a third basis size | `convergence_band.py` | `convergence_band.json` |
| Sec. V C, bunch length: the two modes at 1.6-1.8 ns | `bunch_length_seeded.py` | `bunch_length_seeded.json` |
| Sec. V C, bunch length: l = 0 content | `l0_share.py` | `l0_share.json` |
| Sec. V C: second modes near the most unstable one | `near_degenerate.py` | `near_degenerate.json` |
| Sec. V C, damper: long runs above 0.15 Q_s | `damper_long.py` | `damper_long.json` |
| Sec. V C, damper: resolution and calibration | `damper_trend.py` | `damper_trend.json` |
| Sec. V C, damper: order and number of kick points | `damper_lumped.py` | `damper_lumped.json` |
| Sec. V C, damper: DELPHI at three basis sizes | `delphi_damper_basis.py` | `delphi_damper_basis.json` |
| Sec. V E: DELPHI's parabola on two machines | `psb_delphi_x86.py` | `psb_delphi_x86.json` |
| Sec. V E: Q_r = 100 | `psb_q100.py`, `psb_q100_candidates.py`, `psb_q100_check.py` | `psb_q100*.json` |
| Sec. V F, Table I: DELPHI for the eight bunches | `cb_delphi.py` | `cb_delphi.json` |
| Sec. V F: separation of the coupled-bunch modes | `cb_crosstalk.py` | `cb_crosstalk.json` |

`psb_delphi_x86.py` compares DELPHI's PSB scan on the machine it runs on with the stored `psb.json`; its stored result
was computed on x86 Linux against the Apple M3 values, and it is not part of `run_all.sh`.

Several scripts keep the results they find in their file, so that a long run can be resumed: `linear_limit.py` merges the
configurations it is given into `linear_limit.json`, `resistive_wall.py` keeps the stored simulations and, given slice
counts (`python3 resistive_wall.py 12800 25600`), skips those already stored, and most validation scripts skip the runs
already stored. The others rewrite their file run by run.

`run_all.sh` moves the stored `data/` aside, copies back the one result it does not recompute (`psb_delphi_x86.json`), and
runs everything in an order that respects what each script reads, then `figures.py` and `paper_numbers.py`. It takes weeks
of computing time on one machine (most scripts take hours, `resistive_wall.py` days); the comments in it say which scripts
can run in parallel, memory permitting.

## Results

The scan files in `data/` hold, per simulated point, the tracked growth rate `g` and the predicted one `gp`, and, except in
`resistive_wall.json`, the tracked and predicted real tune shifts `re` and `rep` (all normalized to Q_s). `g` is `null`
for a run that did not grow, except in `psb.json`, `coupled_bunch.json` and `resistive_wall.json`, which store every fitted
rate (negative for the damped coupled-bunch modes); `gp` is 0 and `rep` `null` where no growing mode is predicted, except
below the TMCI threshold, where `rep` is the tune of mode 0. The rows also give the scan parameters (`Qp` for Q', `Nb` in
units of 1e11 in `tmci.json`, `taub` in ns, `damping` in Q_s, `slices`, `nx`, ...), the bucket (`linear`: linearized rf;
`nonlinear`: sinusoidal rf) and, in the single-bunch scans except the resistive wall, how the run was started (`run`:
`seeded`, `kicked`, or, at three points of `bunch_length.json`, `seeded mode -1`, whose kicked runs are kept as `g_kicked`,
`re_kicked`). `psb.json` also gives the predictions with the impedance at the betatron lines (`gp_line`, `rep_line`) and
DELPHI's; `resistive_wall.json` stores the solver scans separately from the simulations; `coupled_bunch.json` stores
whether the smaller basis reproduces each eigenvalue (`held`); `linear_limit.json` holds solver scans only. Most
single-bunch LHC scripts (the tmci, chromaticity, bunch-length, distribution and damper scans and several of the checks)
also save the centroid signal of each run to `data/signals/` (not part of the repository).

## Reproducibility

Every value in `data/` comes from a deterministic solver or from a seeded simulation: numpy's global generator is seeded
before xpart draws the transverse coordinates (`common.matched`, seed 1), the longitudinal samplers use
`numpy.random.default_rng(seed)`, and the coupled-bunch runs start from a quiet bunch. The stored results were computed on
an Apple M3. On another machine the floating-point results differ in the last digits. On x86 Linux, with the same package
versions, the solvers' eigenvalues for the LHC cases agree with the stored ones to within 6e-14 Q_s wherever checked; this
changes the eigenvectors that seed the simulations at the 1e-13 level, and of 25 simulations rerun for up to their first
200 turns, 23 reproduce the stored centroid signals (single precision) bit for bit and the other two agree to 2e-13 and 1.5e-14
relative. The exception is DELPHI's PSB scan with its 39-term Laguerre-series parabola, an ill-conditioned problem: from
Q' = -6 to +4 its eigenvalues differ between the two machines by 1e-12 to 4e-5 Q_s, and at Q' = -8 and -7 its most unstable
mode is a different one, with a growth rate 9% and 0.5% apart (`psb_delphi_x86.json`, Sec. V E of the paper). A different
seed of the macroparticle draw scatters the tracked growth rates by up to about 1% (`multiseed.py`).
