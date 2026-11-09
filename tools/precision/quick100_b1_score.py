"""Score the quick100 B1 scenario of saved ``check_precision.py`` runs with the harness's own functions.

For runs whose batch scenarios are not covered (Wall-OSS's optimized composition takes one request per action call,
so the harness stops at the first batch scenario): the B1 gate (100 windows) only. Uses the harness's
``check_precision.load_run`` / ``Assets`` and ``precision_check.make_plan`` / ``quality`` on the same pairing and its
episode bootstrap; prints, per run, the GT-MSE against the BF16 reference and against the target (old service) run.

    python quick100_b1_score.py <embodied-serving checkout> <assets> <reference> <target> <run> [<run> ...]
"""
import json
import sys
from pathlib import Path

es, assets_dir, ref_dir, target_dir, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
sys.path[:0] = [str(Path(es) / "scripts"), str(Path(es) / "src")]
import numpy as np  # noqa: E402
from check_precision import Assets, load_run  # noqa: E402  (the harness, on the path above)
from embodied_serving.precision_check import make_plan, quality, sha256, variant  # noqa: E402

profile = json.loads((Path(es) / "tests/precision/quick100/profile.json").read_text())
profile["sha256"] = sha256(Path(es) / "tests/precision/quick100/profile.json")
plan = json.loads(json.dumps(make_plan(profile)))
assets = Assets(assets_dir, profile)
ids = [variant(*p) for p in plan["pairs"]]
lookup = {c: i for i, c in enumerate(ids)}
b1 = [n for n, b in plan["batches"].items() if b == 1]
reference, target = load_run(ref_dir), load_run(target_dir)
for run_dir in [target_dir] + runs:
    run = load_run(run_dir)
    assert run["b1_ids"] == ids and run["plan"] == plan and run["checkpoint"] == reference["checkpoint"]
    for name in b1:
        pairs = plan["scenarios"][name]
        idx = [lookup[variant(*p)] for p in pairs]
        delays = [p[1] for p in pairs]
        gt = np.stack([assets.gt[assets.cases[cid]["index"]] for cid, _ in pairs])
        c, r, t = (np.asarray(x["b1"])[idx] for x in (run, reference, target))
        q = quality(c, r, gt, delays)
        vt = quality(c, t, gt, delays, allow_zero=True)
        episodes = [assets.cases[cid]["episode"] for cid, _ in pairs]
        per, clusters = {}, []
        for k, ep in enumerate(dict.fromkeys(episodes)):
            sub = [i for i, e in enumerate(episodes) if e == ep]
            qe = quality(c[sub], r[sub], gt[sub], [delays[i] for i in sub], allow_zero=True)
            per[f"ep{k}"] = round(qe["relative_increase_pct"], 2)
            clusters.append([qe["reference_mse"] * qe["elements"], qe["candidate_mse"] * qe["elements"]])
        clusters = np.array(clusters)
        rng = np.random.default_rng(20260929)  # the harness's bootstrap
        boot = clusters[rng.integers(0, len(clusters), size=(10000, len(clusters)))].sum(axis=1)
        ci = [round(v, 2) for v in np.quantile((boot[:, 1] / boot[:, 0] - 1) * 100, [.025, .975]).tolist()]
        status = "PASS" if q["relative_increase"] <= 0.01 else "FAIL"
        print(json.dumps({"run": Path(run_dir).name, "scenario": name, "windows": q["windows"],
                          "elements": q["elements"], "bf16_mse": q["reference_mse"],
                          "candidate_mse": q["candidate_mse"], "vs_bf16_pct": q["relative_increase_pct"],
                          "status": status, "vs_old_service_pct": vt["relative_increase_pct"], "ci95_pct": ci,
                          "by_episode_vs_bf16_pct": per}))
