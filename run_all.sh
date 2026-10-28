#!/bin/sh
# Recompute every result of the paper, then its figures and numbers. Several scripts keep the runs they find in their result
# file (so that a long run can be resumed), so this starts from an empty data/: the stored results are moved to
# data.stored/ first (compare the two afterwards). Weeks of computing time on one machine. This runs the scripts one after
# the other; each group needs the results of the groups above it, and the scripts within a group can run in parallel,
# except where a comment says otherwise. psb_delphi_x86.py is not run: it compares DELPHI on the machine it runs on with the
# stored psb.json (data/validation/psb_delphi_x86.json holds its x86 Linux result).
#     ./run_all.sh                 # python3 from the PATH
#     PYTHON=/path/to/python ./run_all.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ -d data ]; then
    [ -e data.stored ] && { echo "data.stored/ exists: move it away first"; exit 1; }
    mv data data.stored
    echo "stored results moved to data.stored/"
fi
mkdir -p data/validation logs
run() { echo "$(date '+%F %T') $*"; $PY "$@" > "logs/$(echo "$*" | tr ' ' '_').log" 2>&1; }

# 1. the linear limit and the scans of Table I
run linear_limit.py
run tmci.py
run chromaticity.py
run bunch_length.py
run distributions.py
run damper.py
run resistive_wall.py
run psb.py
run coupled_bunch.py

# 2. the checks of Sec. V that read the scans
run matching.py
run nb_scan.py
run multiseed.py
run bunch_length_seeded.py
run l0_share.py
run damper_long.py
run tmci_seeds.py
run linear_control.py
run near_degenerate.py
run damper_trend.py
run damper_lumped.py B
run damper_lumped.py N2
run damper_lumped.py N4 linear
run damper_lumped.py N4 nonlinear
run delphi_damper_basis.py
run psb_q100.py
run cb_delphi.py
run cb_crosstalk.py

# 3. the checks that read those of group 2
run tmci_mechanism.py
run nb_l0_seeded.py
run damper_lumped.py summary
run psb_q100_check.py
run psb_q100_candidates.py linear
run psb_q100_candidates.py nonlinear
run psb_q100_candidates.py summary                          # after the two above
run convergence_band.py nb_scan --n3 64 80 96               # these two one after the other: both update convergence_band.json
run convergence_band.py chromaticity damper bunch_length distributions matching resistive_wall psb

# 4. the paper
run figures.py
$PY paper_numbers.py > paper_numbers.txt
echo "done: figures/, paper_numbers.txt, logs/"
