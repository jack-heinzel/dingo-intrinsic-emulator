"""Calibration of a trained q1 (run A or B): P-P test on the physical intrinsics and a
joint credible-level test, on injections drawn from the run's own training prior.

    python scripts/eval_calibration.py sample A --n-inj 1000 --n-samples 5000   # GPU
    python scripts/eval_calibration.py plot A B                                 # login node

`sample` writes $OUT/calibration_<run>.pkl with, per injection:
  ranks[i, k]   fraction of posterior samples below the truth in PHYS[k] (run B folded)
  level[i]      fraction of samples with log q > log q(truth): the smallest
                highest-density credible region containing the truth. Uniform if the
                joint density is calibrated (q_fold for run B).
  leak[i]       fraction of samples outside the prior box (flow leakage at the walls)
  snr[i], theta (true parameters)
`plot` makes plots/pp_<runs>.pdf (one panel per run) and prints KS p-values.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from emulator import evaluation as ev  # noqa: E402

OUT = os.environ.get("EVAL_OUT", f"{ev.DATA}/eval")  # override for tests


def sample(args):
    import torch
    from dingo.gw.injection import Injection
    from dingo.gw.noise.asd_dataset import ASDDataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ev.load_model(ev.MODEL_PATH[args.run], device=args.device)
    run = ev.run_of(model)
    assert run == args.run
    inj = Injection.from_posterior_model_metadata(model.metadata)
    inj.asd = ASDDataset(file_name=ev.ASD_PATH)

    res = {k: [] for k in ("ranks", "level", "leak", "snr", "theta")}
    t0 = time.time()
    for i in range(args.n_inj):
        event = inj.random_injection()
        theta = event["parameters"]
        truth = {k: np.array([theta[k]]) for k in ev.PHYS}
        sampler = ev.make_sampler(model, event)
        s = sampler.run_sampler(args.n_samples, batch_size=args.n_samples)
        phys = ev.samples_to_phys(s, run)
        ok = ev.in_bounds(phys)
        res["leak"].append(1 - ok.mean())
        res["ranks"].append([np.mean(phys[k][ok] < truth[k][0]) for k in ev.PHYS])
        # Samples outside the prior box count as below the truth in both runs.
        if run == "A":
            lq_samples = np.where(ok, np.asarray(s["log_prob"], float), -np.inf)
        else:
            lq_samples = np.full(len(ok), -np.inf)
            lq_samples[ok] = ev.log_q_phys(sampler, run, {k: v[ok] for k, v in phys.items()})
        lq_truth = ev.log_q_phys(sampler, run, truth)[0]
        res["level"].append(np.mean(lq_samples > lq_truth))
        res["snr"].append(ev.network_snr(inj, theta, event["asds"]))
        res["theta"].append(theta)
        if i % 50 == 0 or i == args.n_inj - 1:
            print(f"{i + 1}/{args.n_inj}  {time.time() - t0:.0f} s  leak {res['leak'][-1]:.3f}",
                  flush=True)
    res = {k: (np.array(v) if k != "theta" else v) for k, v in res.items()}
    res.update(run=run, params=ev.PHYS, n_samples=args.n_samples, seed=args.seed)
    os.makedirs(OUT, exist_ok=True)
    ev.save(res, f"{OUT}/calibration_{run}{args.tag}.pkl")
    print(f"wrote {OUT}/calibration_{run}{args.tag}.pkl")


LABELS = dict(chirp_mass=r"$\mathcal{M}$", mass_ratio=r"$q$", a_1=r"$a_1$", a_2=r"$a_2$",
              tilt_1=r"$\theta_1$", tilt_2=r"$\theta_2$", luminosity_distance=r"$d_L$",
              level="joint")


def plot(args):
    import matplotlib.pyplot as plt
    from scipy.stats import kstest, combine_pvalues
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import plot_style

    plot_style.apply()
    fig, axes = plt.subplots(1, len(args.runs), figsize=(5.5 * len(args.runs), 5.5), squeeze=False)
    grid = np.linspace(0, 1, 201)
    for ax, run in zip(axes[0], args.runs):
        res = ev.load(f"{OUT}/calibration_{run}{args.tag}.pkl")
        n = len(res["level"])
        # 3-sigma band for the empirical CDF of n uniform draws
        band = 3 * np.sqrt(grid * (1 - grid) / n)
        ax.fill_between(grid, grid - band, grid + band, color="0.85", lw=0)
        cols = {k: res["ranks"][:, j] for j, k in enumerate(res["params"])}
        cols["level"] = res["level"]
        pvals = {}
        for k, x in cols.items():
            pvals[k] = kstest(x, "uniform").pvalue
            ls = "--" if k == "level" else "-"
            ax.plot(grid, [np.mean(x <= g) for g in grid], ls, lw=1.5,
                    label=f"{LABELS[k]} ({pvals[k]:.2f})", color="k" if k == "level" else None)
        comb = combine_pvalues([pvals[k] for k in res["params"]]).pvalue
        print(f"run {run}: n = {n}, KS p-values " + ", ".join(f"{k} {p:.3f}" for k, p in pvals.items())
              + f"; combined (7 marginals) {comb:.3f}; median leak {np.median(res['leak']):.4f}, "
              f"mean leak {np.mean(res['leak']):.4f}")
        ax.set_xlabel("credible level")
        ax.set_ylabel("fraction of injections")
        ax.text(0.04, 0.93, f"run {run}", transform=ax.transAxes)
        ax.legend(loc="lower right", **plot_style.LEG_KW)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plot_style.save(fig, "pp_run" + "".join(args.runs) + args.tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("run", choices=["A", "B", "C"])
    s.add_argument("--n-inj", type=int, default=1000)
    s.add_argument("--n-samples", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="cuda")
    s.add_argument("--tag", default="", help="suffix for the output file, e.g. _seed1")
    q = sub.add_parser("plot")
    q.add_argument("runs", nargs="+")
    q.add_argument("--tag", default="")
    args = p.parse_args()
    {"sample": sample, "plot": plot}[args.cmd](args)
