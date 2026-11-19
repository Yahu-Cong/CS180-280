"""Extract fixed features, train logistic regressions, and evaluate held-out captions."""
import argparse
import csv
import json
import platform
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geometry import FEATURES, PATCHES, disagreement, extract
from select_sample import SEED, SOURCES
from synthetic_check import run as synthetic_check

BOOTSTRAPS = 2000
COUNT_NAMES = ("plane_windows", "raw_grids", "rectified_grids", "matched_grids")


def summarize(gray, depth, K, detector, hog):
    counts, raw, matched = extract(gray, depth, K, detector)
    raw_scores = np.median([entry[0] for entry in raw], axis=0) if raw else np.full(4, np.nan)
    if matched:
        before, after, _ = (np.asarray(v) for v in zip(*matched))
        rectified = np.median(after, axis=0)
        difference = np.median(before - after, axis=0)
        residual = np.median([disagreement(item) for item in after])
        gain = np.median([disagreement(a) - disagreement(b) for a, b in zip(before, after)])
    else:
        rectified = difference = np.full(4, np.nan)
        residual = gain = np.nan
    hog_values = hog.compute(cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)).ravel()
    return counts, raw_scores, rectified, difference, residual, gain, hog_values


def train_predict(features, train, test, labels):
    model = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000))
    return model.fit(features[train], labels[train]).decision_function(features[test])


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def auc_with_bootstrap(scores, weights):
    """AUROC with ties worth half. Give each source the same number of captions."""
    fake = scores[:, 1:].T
    real = scores[:, 0]
    differences = fake[:, :, None] - real
    ranks = (differences > 0) + 0.5 * (differences == 0)
    denominator = np.square(weights.sum(axis=1))
    simulated = np.stack([((weights @ matrix) * weights).sum(axis=1) / denominator
                          for matrix in ranks], axis=1)
    return ranks.mean(axis=(1, 2)), simulated


def auc_summary(estimate, samples):
    return dict(pooled=dict(auc=float(estimate.mean()), ci=interval(samples.mean(axis=1))),
                per_source={source: dict(auc=float(estimate[i]), ci=interval(samples[:, i]))
                            for i, source in enumerate(SOURCES[1:])})


def matched_effect(values, weights, sign):
    """Paired generated-minus-real differences, conditional on both images scoring."""
    effects = {}
    for i, source in enumerate(SOURCES[1:], start=1):
        pair = sign * (values[:, i] - values[:, 0])
        valid = np.isfinite(pair)
        effects[source] = {"pairs": int(valid.sum())}
        if valid.any():
            draws = weights[:, valid]
            den = draws.sum(axis=1)
            means = draws[den > 0] @ pair[valid] / den[den > 0]
            effects[source].update(mean=float(pair[valid].mean()), ci=interval(means))
    return effects


def plot_results(metrics, output):
    models = ("coverage", "geometry", "rgb_grid", "rgb_hog", "rgb_hog_plus_geometry")
    labels = ("Coverage only", "Rectified grid + gain + coverage", "RGB grid",
              "RGB HOG", "RGB HOG + geometry")
    fig, (auc_ax, coverage_ax) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for i, (model, label) in enumerate(zip(models, labels)):
        item = metrics["test_auc"][model]["pooled"]
        lo, hi = item["ci"]
        auc_ax.errorbar(item["auc"], i, xerr=[[item["auc"] - lo], [hi - item["auc"]]],
                        fmt="o", color="#156a78", capsize=4)
    auc_ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    auc_ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Held-out AUROC (95% CI)",
               xlim=(0.35, 1.0), title="Detection performance")
    auc_ax.invert_yaxis()
    coverage_ax.bar(SOURCES, [metrics["coverage"][source]["matched_grid_images"]
                             for source in SOURCES], color="#156a78")
    coverage_ax.tick_params(axis="x", rotation=25)
    coverage_ax.set(ylim=(0, 50), ylabel="Images with a matched grid (of 50)",
                    title="Coverage of the geometry test")
    fig.savefig(output / "results.png", dpi=170)
    plt.close(fig)


def evaluate(sample, predictions, output):
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    with (sample / "manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    hog = cv2.HOGDescriptor((96, 96), (24, 24), (12, 12), (12, 12), 9)
    results = []
    start = time.monotonic()
    for index, row in enumerate(manifest, start=1):
        gray = cv2.imread(str(sample / row["image"]), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (256, 256):
            raise ValueError(f"Expected a 256 × 256 image at {row['image']}")
        with np.load(predictions / f"{row['source']}_{row['caption_id']}.npz") as data:
            depth, K = data["depth"].astype(np.float64), data["intrinsics"].astype(np.float64)
        results.append(summarize(gray, depth, K, detector, hog))
        if index % 150 == 0:
            print(f"Features {index}/{len(manifest)} in {time.monotonic() - start:.1f}s", flush=True)

    counts, raw, rectified, differences, residual, gain, hog_features = map(np.asarray, zip(*results))
    counts = counts / len(PATCHES)
    raw_features = np.column_stack([raw, counts[:, 1]])
    geometry = np.column_stack([rectified, differences, counts])
    inputs = dict(coverage=counts, rgb_grid=raw_features, geometry=geometry,
                  rgb_grid_plus_geometry=np.column_stack([raw_features, geometry]),
                  rgb_hog=hog_features, rgb_hog_plus_geometry=np.column_stack([hog_features, geometry]))
    train = np.array([row["split"] == "train" for row in manifest])
    test = ~train
    sources = np.array([row["source"] for row in manifest])
    labels = sources != "real"
    test_counts = counts[test]
    n = int(test.sum() / len(SOURCES))
    test_rows = [row for row in manifest if row["split"] == "test"]
    for offset in range(0, len(test_rows), len(SOURCES)):
        group = test_rows[offset:offset + len(SOURCES)]
        assert tuple(row["source"] for row in group) == SOURCES
        assert len({row["caption_id"] for row in group}) == 1
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS).astype(float)
    metrics = dict(
        train_counts=dict(Counter(sources[train])), test_counts=dict(Counter(sources[test])),
        software=dict(python=platform.python_version(), numpy=np.__version__,
                      opencv=cv2.__version__, scikit_learn=sklearn.__version__),
        synthetic=synthetic_check(), test_auc={}, leave_one_generator_out={},
        coverage={}, paired_residual=matched_effect(residual[test].reshape(n, -1), weights, 1),
        paired_gain=matched_effect(gain[test].reshape(n, -1), weights, -1),
        rectification_gain={}, incremental_auc={},
    )
    model_scores = {}
    draws = {}
    for name, features in inputs.items():
        scores = train_predict(features, train, test, labels)
        model_scores[name] = scores
        estimate, draws[name] = auc_with_bootstrap(scores.reshape(n, -1), weights)
        metrics["test_auc"][name] = auc_summary(estimate, draws[name])
        leave_out = {}
        for i, source in enumerate(SOURCES[1:], start=1):
            leave_scores = train_predict(features, train & (sources != source), test, labels)
            auc, sims = auc_with_bootstrap(leave_scores.reshape(n, -1)[:, [0, i]], weights)
            leave_out[source] = dict(auc=float(auc[0]), ci=interval(sims[:, 0]))
        metrics["leave_one_generator_out"][name] = leave_out
    for name, baseline in (("rgb_grid_plus_geometry", "rgb_grid"),
                           ("rgb_hog_plus_geometry", "rgb_hog")):
        auc = metrics["test_auc"]
        diff = auc[name]["pooled"]["auc"] - auc[baseline]["pooled"]["auc"]
        metrics["incremental_auc"][name] = dict(delta=diff, ci=interval(
            (draws[name] - draws[baseline]).mean(axis=1)))
    for source in SOURCES:
        selected = sources[test] == source
        metrics["coverage"][source] = {
            key: int((test_counts[selected, j] > 0).sum())
            for j, key in enumerate(("plane_images", "raw_grid_images", "rectified_grid_images",
                                     "matched_grid_images"))
        }
        finite = gain[test][selected]
        finite = finite[np.isfinite(finite)]
        metrics["rectification_gain"][source] = dict(n=int(len(finite)),
            median=float(np.median(finite)) if finite.size else None,
            positive=int((finite > 0).sum()))
    with (output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")
    fields = list(manifest[0]) + list(COUNT_NAMES)
    fields += [f"{prefix}_{name}" for prefix in ("raw", "rectified", "gain") for name in FEATURES]
    fields += ["rectified_residual", "rectification_gain"] + list(model_scores)
    test_indices = {index: j for j, index in enumerate(np.flatnonzero(test))}
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, row in enumerate(manifest):
            values = [*np.rint(counts[i] * len(PATCHES)).astype(int), *raw[i], *rectified[i],
                      *differences[i], residual[i], gain[i]]
            entry = dict(zip(fields, [*row.values(), *values]))
            entry.update({key: "" for key, value in entry.items()
                          if isinstance(value, (float, np.floating)) and not np.isfinite(value)})
            if i in test_indices:
                entry.update({name: scores[test_indices[i]] for name, scores in model_scores.items()})
            writer.writerow(entry)
    plot_results(metrics, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.sample, args.predictions, args.output)
