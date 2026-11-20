"""Baseline: hand-crafted statistical + frequency features -> Random Forest / SVM."""
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(X):
    """X: (N, C, T) -> (N, F). Per channel: mean, std, min, max, median, IQR, energy,
    skewness, kurtosis, dominant frequency bin, spectral entropy. Plus magnitude of
    the first 3 channels and pairwise correlation of those axes."""
    f = [X.mean(2), X.std(2), X.min(2), X.max(2), np.median(X, 2),
         np.percentile(X, 75, axis=2) - np.percentile(X, 25, axis=2),
         (X ** 2).mean(2), stats.skew(X, axis=2), stats.kurtosis(X, axis=2)]
    P = np.abs(np.fft.rfft(X - X.mean(2, keepdims=True), axis=2)) ** 2
    f.append(P[:, :, 1:].argmax(2).astype(np.float32))
    Pn = P / (P.sum(2, keepdims=True) + 1e-12)
    f.append(-(Pn * np.log(Pn + 1e-12)).sum(2))
    mag = np.sqrt((X[:, :3] ** 2).sum(1))
    f += [mag.mean(1, keepdims=True), mag.std(1, keepdims=True)]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        xa, xb = X[:, a] - X[:, a].mean(1, keepdims=True), X[:, b] - X[:, b].mean(1, keepdims=True)
        f.append(((xa * xb).mean(1) / (xa.std(1) * xb.std(1) + 1e-8))[:, None])
    return np.nan_to_num(np.concatenate(f, axis=1).astype(np.float32))


def build_baseline(kind="rf", seed=0, **kw):
    if kind == "rf":
        return make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=kw.get("n_estimators", 300), random_state=seed, n_jobs=-1))
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=kw.get("C", 10.0), probability=True,
                                                   random_state=seed))
    raise ValueError(kind)
