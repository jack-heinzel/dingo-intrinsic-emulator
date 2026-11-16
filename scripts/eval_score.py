"""A score of emulator quality that is comparable across runs with different coordinates and
training priors.

Each run r gives an emulated likelihood L_r = q_r / pi_r (both in the run's own coordinates; the
Jacobian cancels). Moved onto a common prior pi_c (in physical parameters PHYS) it gives the
normalized posterior
    q'_r(theta | d) = L_r(theta) pi_c(theta) / Z_r,
    Z_r = int L_r pi_c dtheta = E_{theta ~ q_r}[pi_c(theta) / pi_r(theta)],
estimated by importance sampling from the run's own posterior (samples outside the prior box
weigh 0, so leakage is penalized). The score is the log density at the truth, on injections
drawn from pi_c:
    s_r = log q_r,phys(theta*) - log pi_r,phys(theta*) + log pi_c(theta*) - log Z_r.
s is a proper scoring rule: E[s_r] = -H_c(theta | d) - E_d KL(p_c(. | d) || q'_r), and the first
term is the same for all runs on the same injections. So differences of mean s between runs,
paired by injection, are differences of the expected KL. The training loss can't be compared
like this: it carries each run's own posterior entropy, coordinates and standardization.

Heavy tail: pi_c / pi_r grows like 1/a at zero spin when pi_c is uniform in a (A's prior)
and pi_r is not (B: p(a) ~ a; D: p(a) ~ a/a0 below a0). The weights then have a
logarithmically divergent second moment. We record the ESS and a bootstrap error of log Z per
injection.

    python scripts/eval_score.py sample --runs A B D --n-inj 1000 --n-samples 20000   # GPU
    python scripts/eval_score.py summary                                             # login
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from emulator import evaluation as ev  # noqa: E402

OUT = os.environ.get("EVAL_OUT", f"{ev.DATA}/eval")


def _logmeanexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(x - m), axis=axis))


def score_event(sampler, run, prior_r, prior_c, event, n_samples, rng, n_boot=200):
    theta = event["parameters"]
    truth = {k: np.array([theta[k]], float) for k in ev.PHYS}
    # truth: run density in physical parameters, then reweight to pi_c
    lq_truth = (ev.log_q_phys(sampler, run, truth) + ev.log_jacobian_phys(run, truth))[0]
    lw_truth = (ev.log_prior_phys(prior_c, truth) - ev.log_prior_phys(prior_r, truth))[0]
    # normalization Z_r from the run's own samples
    s = sampler.run_sampler(n_samples, batch_size=min(n_samples, 20000))
    phys = ev.samples_to_phys(s, run)
    ok = ev.in_bounds(phys)
    lw = np.full(len(ok), -np.inf)
    sub = {k: v[ok] for k, v in phys.items()}
    lw[ok] = ev.log_prior_phys(prior_c, sub) - ev.log_prior_phys(prior_r, sub)
    log_z = _logmeanexp(lw)
    w = np.exp(lw - np.max(lw[ok]))
    ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
    boot = _logmeanexp(lw[rng.integers(0, len(lw), (n_boot, len(lw)))], axis=1)
    return dict(score=lq_truth + lw_truth - log_z, log_q_truth=lq_truth, log_w_truth=lw_truth,
                log_z=log_z, log_z_err=np.std(boot), ess=ess, leak=1 - ok.mean())


def sample(args):
    import torch
    from dingo.gw.injection import Injection
    from dingo.gw.inference.context import GWSamplerContext
    from dingo.gw.noise.asd_dataset import ASDDataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    md_c = ev.load_metadata(ev.MODEL_PATH[args.prior_run])
    inj = Injection.from_posterior_model_metadata(md_c)
    inj.asd = ASDDataset(file_name=ev.ASD_PATH)
    events = [inj.random_injection() for _ in range(args.n_inj)]
    snr = np.array([ev.network_snr(inj, e["parameters"], e["asds"]) for e in events])
    print(f"{len(events)} injections from run {args.prior_run}'s prior; median SNR {np.median(snr):.1f}",
          flush=True)
    prior_c = GWSamplerContext.from_model_metadata(
        md_c, {"waveform": events[0]["waveform"], "asds": events[0]["asds"]}).prior

    rng = np.random.default_rng(args.seed)
    out = dict(prior_run=args.prior_run, seed=args.seed, n_samples=args.n_samples, snr=snr,
               theta=[e["parameters"] for e in events], runs={})
    path = f"{OUT}/score_prior{args.prior_run}{args.tag}.pkl"
    for run in args.runs:
        t0 = time.time()
        model = ev.load_model(ev.MODEL_PATH[run], device=args.device)
        assert ev.run_of(model) == run
        rows = []
        for i, event in enumerate(events):
            sampler = ev.make_sampler(model, event)
            rows.append(score_event(sampler, run, sampler.context.prior, prior_c, event,
                                    args.n_samples, rng))
            if i % 100 == 0 or i == len(events) - 1:
                print(f"run {run}: {i + 1}/{len(events)}  {time.time() - t0:.0f} s  "
                      f"mean score {np.mean([r['score'] for r in rows]):.3f}", flush=True)
        out["runs"][run] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        del model
        torch.cuda.empty_cache()
        os.makedirs(OUT, exist_ok=True)
        ev.save(out, path)  # after each run, so a timeout keeps the finished ones
    print(f"wrote {path}")


def summary(args):
    d = ev.load(f"{OUT}/score_prior{args.prior_run}{args.tag}.pkl")
    runs = list(d["runs"])
    ref = args.ref if args.ref in runs else runs[0]
    snr = d["snr"]
    bins = [(0, 8), (8, 15), (15, 30), (30, np.inf)]
    print(f"{len(snr)} injections from run {d['prior_run']}'s prior, {d['n_samples']} samples each.\n"
          f"Score = log q'(truth); higher is better. Differences vs run {ref}, paired, +- standard error.")
    head = f"{'run':>4} {'mean score':>11} {'diff vs ' + ref:>14} " + " ".join(
        f"{f'SNR {lo:g}-{hi:g}':>15}" for lo, hi in bins) + f" {'med ESS':>8} {'max dlogZ':>9} {'leak':>6}"
    print(head)
    s_ref = d["runs"][ref]["score"]
    for r in runs:
        x = d["runs"][r]
        diff = x["score"] - s_ref
        cells = []
        for lo, hi in bins:
            m = (snr >= lo) & (snr < hi)
            cells.append(f"{diff[m].mean():+7.3f}+-{diff[m].std() / np.sqrt(max(m.sum(), 1)):.3f}"
                         if m.sum() > 1 else f"{'-':>15}")
        print(f"{r:>4} {x['score'].mean():11.3f} {diff.mean():+8.3f}+-{diff.std() / np.sqrt(len(diff)):.3f} "
              + " ".join(f"{c:>15}" for c in cells)
              + f" {np.median(x['ess']):8.3f} {np.max(x['log_z_err']):9.3f} {np.median(x['leak']):6.4f}")
    print("SNR bins hold " + ", ".join(f"{((snr >= lo) & (snr < hi)).sum()}" for lo, hi in bins)
          + " injections.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--runs", nargs="+", choices=ev.RUNS, required=True)
    s.add_argument("--prior-run", default="A", help="run whose training prior is pi_c and draws injections")
    s.add_argument("--n-inj", type=int, default=1000)
    s.add_argument("--n-samples", type=int, default=20000)
    s.add_argument("--seed", type=int, default=2)
    s.add_argument("--device", default="cuda")
    s.add_argument("--tag", default="")
    q = sub.add_parser("summary")
    q.add_argument("--prior-run", default="A")
    q.add_argument("--tag", default="")
    q.add_argument("--ref", default="A")
    args = p.parse_args()
    {"sample": sample, "summary": summary}[args.cmd](args)
