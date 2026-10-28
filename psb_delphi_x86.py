"""PSB paragraph of Sec. V E: DELPHI's 39-term Laguerre-series parabola makes its PSB eigenvalue problem ill-conditioned at
the most negative chromaticities, where its growth rates "differ between machines". psb.delphi (DELPHI at (13,40), most
unstable dQ/Q_s) at every Q' of the PSB scan, computed on this machine and compared with the values of data/psb.json
('delphi', computed on an Apple M3). The stored file is the comparison made on an x86 Linux machine.
Writes data/validation/psb_delphi_x86.json; a few minutes per Q'.

    python3 psb_delphi_x86.py
"""
import os, json, platform, time
import numpy as np

import common as c
import psb

OUT = os.path.join(c.DATA, 'validation')
PATH = os.path.join(OUT, 'psb_delphi_x86.json')

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    stored = {d['Qp']: [d['re'], d['g']] for d in c.load('psb')['delphi']}
    rows = []
    for Qp in psb.Qpscan:
        t0 = time.time()
        d = psb.delphi(Qp)
        this, ref = [d.real, -d.imag], stored[float(Qp)]
        rows.append(dict(Qp=float(Qp), this_machine=this, psb_json=ref, re_diff=this[0] - ref[0], g_rel_diff=this[1]/ref[1] - 1))
        print(f"Q' = {Qp:+.0f}: this machine {this[0]:+.6f}/{this[1]:.6f}, psb.json {ref[0]:+.6f}/{ref[1]:.6f}, growth {this[1]/ref[1] - 1:+.2e} "
              f"[{(time.time() - t0)/60:.1f} min]", flush=True)
    out = dict(what="DELPHI's most unstable PSB eigenvalue (Re dQ/Qs, growth/Qs), psb.delphi at (13,40), on this machine ('this_machine') "
                    "and in data/psb.json ('psb_json', computed on an Apple M3)",
               machine=dict(platform=platform.platform(), processor=platform.machine(), python=platform.python_version(), numpy=np.__version__),
               rows=rows)
    with open(PATH, 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
    print('done')
