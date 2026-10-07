"""Compare SEAL to two baselines on the same objective (trajectory coherence rho,
rule complexity c, combined cost rho + lambda*c) on the four public panels.

  SEAL                 : joint search; read from results/<ds>/ produced by run_panels.py.
  Cluster-then-Explain : k-means on the pyramid representation (K chosen by silhouette),
                         then the same shallow tree on the descriptors to describe it.
  EMM                  : exceptional model mining with a mean-trajectory model; a beam
                         search over descriptor conjunctions, each subgroup scored by its
                         size-weighted squared deviation from the population mean.

Run run_panels.py first so the SEAL results exist, then this script.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier

from seal import pyramid_features, tree_to_rules
from run_panels import load, PANELS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LAM = 0.005


def within_ratio(labels, dist, gm):
    labels = np.asarray(labels)
    blocks = []
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if len(idx) > 1:
            blocks.append(dist[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])
    return float(np.concatenate(blocks).mean() / gm) if blocks else np.inf


def complexity_from_text(text):
    total = 0
    for seg in text.split("|"):
        seg = seg.strip()
        if seg.startswith("default") or not seg:
            continue
        body = re.sub(r"^G\d+:\s*", "", seg)
        for clause in re.split(r"\)\s*OR\s*\(", body.strip()):
            total += len([a for a in re.split(r"\s+AND\s+", clause.strip().strip("()")) if a.strip()])
    return total


def cluster_then_explain(X, dist, gm, rep):
    N = len(X); Xv = X.to_numpy()
    best_sil, Cbest = -2, None
    for K in range(2, 9):
        C = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(rep)
        s = silhouette_score(rep, C)
        if s > best_sil:
            best_sil, Cbest = s, C
    leaf = max(int(0.02 * N), 5)
    best = None
    for depth in (1, 2, 3, 4):
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(Xv, Cbest)
        pred = tree.predict(Xv)
        rules = tree_to_rules(tree, list(X.columns))
        default = max(rules, key=lambda c: int((pred == c).sum()))
        c = sum(len(p) for cc, paths in rules.items() if cc != default for p in paths)
        rho = within_ratio(pred, dist, gm)
        if best is None or rho + LAM * c < best[0]:
            best = (rho + LAM * c, rho, c, len(np.unique(pred)))
    return best[1], best[2], best[3]


def emm(X, traj, dist, gm, K, max_depth=4, beam=40, n_thr=7):
    N = len(X); Xv = X.to_numpy(); Y = traj.to_numpy().astype(float); ybar = Y.mean(0)
    atoms = []
    for j in range(Xv.shape[1]):
        col = Xv[:, j]
        if len(np.unique(col)) <= 2:
            atoms.append((j, ">", 0.5))
        else:
            for t in np.unique(np.round(np.quantile(col, np.linspace(0.1, 0.9, n_thr)), 6)):
                atoms.append((j, "<=", float(t))); atoms.append((j, ">", float(t)))

    def mask(conds):
        m = np.ones(N, bool)
        for (j, op, t) in conds:
            m &= (Xv[:, j] <= t) if op == "<=" else (Xv[:, j] > t)
        return m

    def qual(m):
        n = int(m.sum())
        if n < 5 or n == N:
            return -1.0
        dev = Y[m].mean(0) - ybar
        return (n / N) * float(dev @ dev)

    beamset = sorted([([a], mask([a])) for a in atoms], key=lambda cm: qual(cm[1]), reverse=True)[:beam]
    found = list(beamset)
    for _ in range(2, max_depth + 1):
        cand = [(conds + [a], mask(conds + [a])) for conds, _m in beamset for a in atoms]
        beamset = sorted(cand, key=lambda cm: qual(cm[1]), reverse=True)[:beam]
        found += beamset
    found = sorted(found, key=lambda cm: qual(cm[1]), reverse=True)

    def jac(a, b):
        return (a & b).sum() / max((a | b).sum(), 1)
    selected = []
    for conds, m in found:
        if any(jac(m, sm) > 0.5 for _, sm in selected):
            continue
        selected.append((conds, m))
        if len(selected) >= K:
            break
    labels = np.zeros(N, int)
    for i, (conds, m) in enumerate(selected):
        labels[(labels == 0) & m] = i + 1
    c = sum(len(conds) for conds, _ in selected)
    return within_ratio(labels, dist, gm), c, len(np.unique(labels))


if __name__ == "__main__":
    print(f"{'panel':10s} | {'method':20s} | {'rho':>6s} {'c':>3s} {'cost':>7s} {'grp':>3s}")
    print("-" * 60)
    for ds in PANELS:
        traj, X = load(ds)
        rep = pyramid_features(traj.to_numpy().astype(float))
        dist = squareform(pdist(rep)); gm = float(dist[np.triu_indices(len(traj), k=1)].mean())

        lab = pd.read_csv(RESULTS / ds / "membership.csv").set_index("entity")["subgroup"].reindex(traj.index).to_numpy()
        rho_s = within_ratio(lab, dist, gm)
        c_s = complexity_from_text((RESULTS / ds / "rules.txt").read_text(encoding="utf-8"))
        ng_s = len(np.unique(lab))

        rho_b, c_b, ng_b = cluster_then_explain(X, dist, gm, rep)
        rho_e, c_e, ng_e = emm(X, traj, dist, gm, K=ng_s - 1)

        for name, rho, c, ng in [("SEAL", rho_s, c_s, ng_s),
                                 ("Cluster-then-Explain", rho_b, c_b, ng_b),
                                 ("EMM", rho_e, c_e, ng_e)]:
            print(f"{ds:10s} | {name:20s} | {rho:6.3f} {c:3d} {rho + LAM * c:7.3f} {ng:3d}")
        print("-" * 60)
