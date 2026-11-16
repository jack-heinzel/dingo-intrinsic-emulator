"""Accuracy of the emulated marginal likelihood against a direct calculation.

For each injection and each intrinsic point theta_int, the reference is
    log L_ref(theta_int) = log of the integral of L(d | theta_int, ext) pi(ext) d ext
over ext = (ra, dec, psi, theta_jn, phase, phi_12, phi_jl), with geocent_time
marginalized analytically (FFT, n_fft = 16; converged to 1e-3) and the training priors.
The 7D integral is a dynesty evidence. Data, waveforms and likelihood are dingo's own
(GWSamplerContext.likelihood), so they match what the networks were trained on.

    python scripts/eval_reference.py prepare --n-events 20 --n-points 30    # GPU
    python scripts/eval_reference.py reference --nproc 128                  # CPU node
    python scripts/eval_reference.py analyze                                # login node

prepare: injections from run A's (stock) prior with network SNR >= --snr-min, a random
O3 ASD each. Points: the truth, plus n_points/2 posterior draws from each of A and B
(folded, inside the prior), so neither run is favoured. Stores both emulators' log L.
reference: one dynesty run per point, run in parallel; resumable (one file per point).
analyze: residual r = log L_emu - log L_ref, minus its mean over each event's points (the
emulator is defined up to a constant per event). Prints the spread for A and B and
plots residuals against log L_ref.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from emulator import evaluation as ev  # noqa: E402

OUT = os.environ.get("EVAL_OUT", f"{ev.DATA}/eval/reference")  # override for tests
N_FFT = 16


def prepare(args):
    import torch
    from dingo.gw.injection import Injection
    from dingo.gw.noise.asd_dataset import ASDDataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    models = {r: ev.load_model(ev.MODEL_PATH[r], device=args.device) for r in args.runs}
    # injections from the prior of run args.inj_run (all runs share domain and detectors)
    inj = Injection.from_posterior_model_metadata(ev.load_metadata(ev.MODEL_PATH[args.inj_run]))
    inj.asd = ASDDataset(file_name=ev.ASD_PATH)

    events, n_tried = [], 0
    while len(events) < args.n_events:
        n_tried += 1
        event = inj.random_injection()
        theta = event["parameters"]
        snr = ev.network_snr(inj, theta, event["asds"])
        if snr < args.snr_min:
            continue
        samplers = {r: ev.make_sampler(m, event) for r, m in models.items()}
        pts = {k: [np.array([theta[k]])] for k in ev.PHYS}
        source = [np.array(["truth"])]
        for r, smp in samplers.items():
            s = smp.run_sampler(20 * args.n_points)
            phys = ev.samples_to_phys(s, r)
            idx = np.flatnonzero(ev.in_bounds(phys))[: args.n_points // 2]
            for k in ev.PHYS:
                pts[k].append(phys[k][idx])
            source.append(np.full(len(idx), r))
        pts = {k: np.concatenate(v) for k, v in pts.items()}
        rec = dict(waveform=event["waveform"], asds=event["asds"], theta=theta, snr=snr,
                   points=pts, source=np.concatenate(source))
        for r, smp in samplers.items():
            rec[f"logL_{r}"] = ev.log_L_emulator(smp, r, pts)
        events.append(rec)
        print(f"event {len(events)}/{args.n_events} (tried {n_tried}): SNR {snr:.1f}, "
              f"{len(pts['chirp_mass'])} points", flush=True)
    os.makedirs(OUT, exist_ok=True)
    ev.save(dict(events=events, metadata=models[args.runs[0]].metadata, runs=args.runs, args=vars(args)),
            f"{OUT}/events.pkl")
    print(f"wrote {OUT}/events.pkl")


# --- reference evidences (CPU) ------------------------------------------------------------

_CACHE = {}


def _likelihood(i):
    """Per-process cache of the time-marginalized likelihood for event i."""
    if i not in _CACHE:
        from dingo.gw.inference.context import GWSamplerContext
        e = _EVENTS[i]
        ctx = GWSamplerContext.from_model_metadata(_META, {"waveform": e["waveform"], "asds": e["asds"]})
        lik = ctx.likelihood(time_marginalization_kwargs=dict(t_lower=-0.1, t_upper=0.1, n_fft=N_FFT))
        _CACHE.clear()  # one event per process at a time keeps memory small
        _CACHE[i] = (lik, [ctx.prior[k] for k in ev.EXTRINSIC])
    return _CACHE[i]


def _evidence(task):
    import dynesty
    i, j, nlive, seed = task
    path = f"{OUT}/points/ev{i:03d}_pt{j:03d}.npy"
    if os.path.exists(path):
        return path
    lik, priors = _likelihood(i)
    fixed = {k: float(_EVENTS[i]["points"][k][j]) for k in ev.PHYS}
    fixed["geocent_time"] = 0.0  # marginalized

    def ptf(u):
        return np.array([p.rescale(x) for p, x in zip(priors, u)])

    def logl(x):
        return lik.log_likelihood({**fixed, **dict(zip(ev.EXTRINSIC, x))})

    t = time.time()
    s = dynesty.NestedSampler(logl, ptf, len(ev.EXTRINSIC), nlive=nlive, bound="multi",
                              sample="rwalk", rstate=np.random.default_rng(seed))
    s.run_nested(dlogz=0.1, print_progress=False)
    r = s.results
    np.save(path, np.array([r.logz[-1], r.logzerr[-1], np.sum(r.ncall), time.time() - t]))
    return path


def _init(events_path):
    global _EVENTS, _META
    d = ev.load(events_path)
    _EVENTS, _META = d["events"], d["metadata"]


def reference(args):
    from multiprocessing import Pool
    os.makedirs(f"{OUT}/points", exist_ok=True)
    _init(f"{OUT}/events.pkl")
    tasks = [(i, j, args.nlive, 1000 * i + j) for i, e in enumerate(_EVENTS)
             for j in range(len(e["points"]["chirp_mass"]))
             if not os.path.exists(f"{OUT}/points/ev{i:03d}_pt{j:03d}.npy")]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} evidences to compute on {args.nproc} processes", flush=True)
    t0 = time.time()
    # chunksize 1 and tasks ordered by event: each worker mostly stays on one event
    with Pool(args.nproc, initializer=_init, initargs=(f"{OUT}/events.pkl",)) as pool:
        for n, _ in enumerate(pool.imap_unordered(_evidence, tasks, chunksize=1), 1):
            if n % 25 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)}  {time.time() - t0:.0f} s", flush=True)


# --- analysis -----------------------------------------------------------------------------

def analyze(args):
    d = ev.load(f"{OUT}/events.pkl")
    runs = d.get("runs", "AB")
    rows = []
    for i, e in enumerate(d["events"]):
        n = len(e["points"]["chirp_mass"])
        ref = np.full((n, 4), np.nan)
        for j in range(n):
            f = f"{OUT}/points/ev{i:03d}_pt{j:03d}.npy"
            if os.path.exists(f):
                ref[j] = np.load(f)
        ok = np.isfinite(ref[:, 0])
        if ok.sum() < 3:
            continue
        rec = dict(i=i, snr=e["snr"], logZ=ref[ok, 0], err=ref[ok, 1], source=e["source"][ok])
        for r in runs:
            res = e[f"logL_{r}"][ok] - ref[ok, 0]
            rec[f"res_{r}"] = res - np.mean(res)
        rows.append(rec)
    if not rows:
        print("no finished events")
        return
    print(f"{len(rows)} events with reference evidences; dynesty error median "
          f"{np.median(np.concatenate([r['err'] for r in rows])):.3f}")
    print(" ev   SNR   n   " + "  ".join(f"std(res_{run})" for run in runs))
    for r in rows:
        print(f"{r['i']:3d} {r['snr']:5.1f} {len(r['logZ']):3d}   "
              + "  ".join(f"{np.std(r[f'res_{run}']):9.3f}" for run in runs))
    all_err2 = np.mean(np.concatenate([r["err"] for r in rows]) ** 2)
    for run in runs:
        res = np.concatenate([r[f"res_{run}"] for r in rows])
        print(f"run {run}: RMS residual {np.sqrt(np.mean(res ** 2)):.3f} "
              f"(reference noise alone ~{np.sqrt(all_err2):.3f}); median |res| {np.median(np.abs(res)):.3f}; "
              f"95% |res| {np.quantile(np.abs(res), 0.95):.3f}")

    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for run, c in zip(runs, [f"C{i}" for i in range(10)]):
        x = np.concatenate([r["logZ"] - r["logZ"].max() for r in rows])
        y = np.concatenate([r[f"res_{run}"] for r in rows])
        ax.scatter(x, y, s=8, color=c, alpha=0.6, lw=0, label=f"run {run}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\ln L_{\rm ref} - \max \ln L_{\rm ref}$ (per event)")
    ax.set_ylabel(r"$\ln L_{\rm emu} - \ln L_{\rm ref}$ (event mean removed)")
    ax.legend(**plot_style.LEG_KW)
    plot_style.save(fig, "reference_residuals_run" + "".join(runs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--n-events", type=int, default=20)
    s.add_argument("--n-points", type=int, default=30)
    s.add_argument("--snr-min", type=float, default=10.0)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--device", default="cuda")
    s.add_argument("--runs", default="AB", help="runs to evaluate, e.g. ABC")
    s.add_argument("--inj-run", default="A", help="run whose training prior draws the injections")
    s = sub.add_parser("reference")
    s.add_argument("--nproc", type=int, default=os.cpu_count())
    s.add_argument("--nlive", type=int, default=500)
    s.add_argument("--limit", type=int, default=0, help="only this many evidences (testing)")
    sub.add_parser("analyze")
    args = p.parse_args()
    {"prepare": prepare, "reference": reference, "analyze": analyze}[args.cmd](args)
