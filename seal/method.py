"""SEAL: interpretable trajectory segmentation for longitudinal data.

SEAL groups entities by the shape of their target trajectory and describes each
group with a compact rule over the entity descriptors. A single randomized search
fits the grouping and its rule together and scores each candidate by

    cost = rho + lambda * complexity,

where rho is the within-group trajectory distance relative to the panel average and
complexity counts the propositions in the rules. Only numpy, pandas, scipy and
scikit-learn are required.

Pipeline:
  - multi-resolution pyramid representation of each trajectory, Euclidean distance
  - randomized search over clustering (k-means / agglomerative) -> decision tree,
    scored by the cost above
  - tree read as a first-match rule set
  - diversity ranking of the strong candidates by adjusted Rand index
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier


# --- trajectory representation and distance -------------------------------- #
def pyramid_features(Y: np.ndarray, levels: int = 6) -> np.ndarray:
    """Multi-resolution (Haar) pyramid of each trajectory: the sum over the whole
    series, the sums over its two halves, its four quarters, and so on for up to
    `levels` levels. If a series is too short for the finest level, the pyramid
    stops early and the raw series is appended as the finest level."""
    T = Y.shape[1]
    blocks, truncated = [], False
    for level in range(levels):
        n_seg = 2 ** level
        if n_seg > T:
            truncated = True
            break
        blocks.append(np.stack([seg.sum(axis=1) for seg in np.array_split(Y, n_seg, axis=1)], axis=1))
    if truncated:
        blocks.append(Y.astype(float))
    return np.concatenate(blocks, axis=1)


def build_distance(traj: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """traj: wide (entities x time) target matrix, sorted by entity. Returns the
    pyramid representation, the N x N distance matrix, and the global mean distance."""
    rep = pyramid_features(traj.to_numpy().astype(float))
    dist = squareform(pdist(rep))
    n = len(traj)
    global_mean = dist[np.triu_indices(n, k=1)].mean()
    return rep, dist, float(global_mean)


# --- tree -> readable rules ------------------------------------------------- #
def tree_to_rules(tree, feature_names):
    """Read a fitted tree as {class: [path, ...]}; a path is a list of
    (feature, op, threshold), repeated tests on one feature merged to the
    tightest interval."""
    t = tree.tree_
    raw_paths = []

    def subtree_class(node):
        if t.children_left[node] == -1:
            return int(np.argmax(t.value[node]))
        a = subtree_class(t.children_left[node])
        b = subtree_class(t.children_right[node])
        return a if (a is not None and a == b) else None

    def walk(node, conds):
        pure = subtree_class(node)
        if pure is not None:
            raw_paths.append((pure, conds))
            return
        feat, thr = feature_names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], conds + [(feat, "<=", thr)])
        walk(t.children_right[node], conds + [(feat, ">", thr)])

    walk(0, [])
    rules = {}
    for cls_idx, conds in raw_paths:
        bounds = {}
        for feat, op, thr in conds:
            lo, hi = bounds.get(feat, (-np.inf, np.inf))
            bounds[feat] = (max(lo, thr), hi) if op == ">" else (lo, min(hi, thr))
        path = [(f, ">", lo) for f, (lo, hi) in bounds.items() if lo > -np.inf] + \
               [(f, "<=", hi) for f, (lo, hi) in bounds.items() if hi < np.inf]
        rules.setdefault(int(tree.classes_[cls_idx]), []).append(path)
    return rules


def rules_to_text(rules, default):
    parts = []
    for label, paths in rules.items():
        if label == default:
            continue
        conj = [" AND ".join(f"{f} {op} {thr:.4g}" for f, op, thr in p) for p in paths]
        parts.append(f"G{label}: " + (" OR ".join(f"({c})" for c in conj) if len(conj) > 1 else conj[0]))
    return " | ".join(parts + ["default: TRUE"])


def apply_rules(Xdf, rules, default):
    """First-match rule evaluation (reproduces the tree partition)."""
    lab = np.full(len(Xdf), default, dtype=int)
    assigned = np.zeros(len(Xdf), dtype=bool)
    for label, paths in rules.items():
        if label == default:
            continue
        fires = np.zeros(len(Xdf), dtype=bool)
        for path in paths:
            ok = np.ones(len(Xdf), dtype=bool)
            for f, op, thr in path:
                col = Xdf[f].to_numpy()
                ok &= (col > thr) if op == ">" else (col <= thr)
            fires |= ok
        lab[fires & ~assigned] = label
        assigned |= fires
    return lab


# --- randomized clustering -> tree search ---------------------------------- #
def search(rep, dist, global_mean, X, *, n_candidates, k_range, tree_depths,
           leaf_fractions, lam, seed):
    """Randomized search for the best rule set. Returns (records, best, trees,
    all_labels)."""
    entities = X.index.to_numpy()

    def within_ratio(labels):
        pair_blocks = []
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            if len(idx) > 1:
                pair_blocks.append(dist[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])
        return np.concatenate(pair_blocks).mean() / global_mean if pair_blocks else np.inf

    rng = np.random.default_rng(seed)
    Xv = X.to_numpy()
    records, trees, all_labels, best = [], [], [], None
    for _ in range(n_candidates):
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        seed_i = int(rng.integers(2**31 - 1))
        if rng.random() < 0.5:
            clus = KMeans(n_clusters=k, n_init=1, random_state=seed_i).fit_predict(rep)
        else:
            clus = AgglomerativeClustering(
                n_clusters=k, linkage=str(rng.choice(["ward", "complete", "average"]))
            ).fit_predict(rep)
        leaf = max(int(rng.uniform(*leaf_fractions) * len(entities)), 5)
        tree = DecisionTreeClassifier(
            max_depth=int(rng.choice(tree_depths)), min_samples_leaf=leaf,
            max_features=[None, 0.5][int(rng.random() < 0.5)], random_state=seed_i,
        ).fit(Xv, clus)
        trees.append(tree)
        labels = tree.predict(Xv)
        all_labels.append(labels.astype(np.int8))
        rules = tree_to_rules(tree, list(X.columns))
        default = max(rules, key=lambda c: int((labels == c).sum()))
        complexity = sum(len(p) for c, paths in rules.items() if c != default for p in paths)
        ratio = within_ratio(labels)
        cost = ratio + lam * complexity
        records.append({"cost": cost, "distance_ratio": ratio, "complexity": complexity,
                        "n_groups": len(rules), "rules": rules_to_text(rules, default)})
        if best is None or cost < best["cost"]:
            best = {**records[-1], "labels": labels, "rules_struct": rules,
                    "default": default, "tree": tree}
    return records, best, trees, all_labels


def diversity_ranking(records, all_labels, *, pool, ari_max):
    """Greedy shortlist whose labelings stay far apart under the adjusted Rand index."""
    order = np.argsort([r["cost"] for r in records], kind="stable")[:pool]
    sel_rows, sel_labels = [], []
    for cand in order:
        lab_c = all_labels[int(cand)]
        max_ari = max((adjusted_rand_score(s, lab_c) for s in sel_labels), default=None)
        if max_ari is None or max_ari <= ari_max:
            sel_rows.append({"candidate": int(cand), **records[int(cand)], "max_ari_to_selected": max_ari})
            sel_labels.append(lab_c)
    return sel_rows


def run_seal(traj: pd.DataFrame, X: pd.DataFrame, params: dict) -> dict:
    """Run SEAL end to end on a wide target-trajectory matrix `traj`
    (entities x time) and a descriptor matrix `X` (entities x features, same entity
    order). Returns the best rule set, its labels, all records, and the shortlist."""
    assert list(traj.index) == list(X.index), "traj and X must share entity order"
    rep, dist, global_mean = build_distance(traj)
    records, best, trees, all_labels = search(
        rep, dist, global_mean, X,
        n_candidates=params["n_candidates"], k_range=params["k_range"],
        tree_depths=params["tree_depths"], leaf_fractions=params["leaf_fractions"],
        lam=params["lambda_complexity"], seed=params["seed"],
    )
    diverse = diversity_ranking(records, all_labels, pool=params["diversity_pool"], ari_max=params["ari_max"])
    return {
        "entities": traj.index.to_numpy(),
        "best": best,
        "records": records,
        "diverse": diverse,
        "diverse_trees": [trees[int(r["candidate"])] for r in diverse],
        "global_mean_distance": global_mean,
    }
