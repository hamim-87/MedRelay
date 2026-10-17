"""Model A - clinical action classification (multi-label over 8 action categories + no_action).

Baselines: TF-IDF + logistic regression, TF-IDF + linear SVM (sigmoid-calibrated margins), optional compact neural net (TF-IDF -> SVD -> MLP).
Training labels are SILVER (rule-proposed) labels from the dataset pipeline; evaluation on them measures consistency, not clinical accuracy.
`evaluate_on_spans` additionally scores the classifier on developer-annotated obligation spans (src/eval_sets.py), a separate label source.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .evaluation import multilabel_report, single_label_confusion
from .ontology import ALL_LABELS, CATEGORIES


def labels_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"y_{c}" for c in ALL_LABELS]].values.astype(int)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class ActionClassifier:
    def __init__(self, kind: str = "logreg", C: float = 2.0, word_ngrams=(1, 2), char: bool = True, class_weight="balanced", seed: int = 0, mlp_hidden: int = 128):
        self.kind, self.C, self.word_ngrams, self.char, self.class_weight, self.seed, self.mlp_hidden = kind, C, tuple(word_ngrams), char, class_weight, seed, mlp_hidden
        self.labels = ALL_LABELS
        self.params = dict(kind=kind, C=C, word_ngrams=list(word_ngrams), char=char, class_weight=class_weight, seed=seed)

    def _vec_fit(self, texts):
        self.v_word = TfidfVectorizer(ngram_range=self.word_ngrams, min_df=2, sublinear_tf=True, lowercase=True)
        mats = [self.v_word.fit_transform(texts)]
        if self.char:
            self.v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)
            mats.append(self.v_char.fit_transform(texts))
        return hstack(mats).tocsr()

    def _vec(self, texts):
        mats = [self.v_word.transform(texts)]
        if self.char:
            mats.append(self.v_char.transform(texts))
        return hstack(mats).tocsr()

    def fit(self, texts, Y):
        t = time.perf_counter()
        X = self._vec_fit(list(texts))
        Y = np.asarray(Y)
        self.keep = [j for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
        if self.kind == "logreg":
            self.clf = OneVsRestClassifier(LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=2000, solver="liblinear", random_state=self.seed))
        elif self.kind == "svm":
            self.clf = OneVsRestClassifier(LinearSVC(C=self.C, class_weight=self.class_weight, dual=True, max_iter=5000, random_state=self.seed))
        elif self.kind == "mlp":
            self.svd = TruncatedSVD(n_components=200, random_state=self.seed)
            X = self.svd.fit_transform(X)
            self.clf = MLPClassifier(hidden_layer_sizes=(self.mlp_hidden,), alpha=1e-3, max_iter=60, early_stopping=True, random_state=self.seed)
        else:
            raise ValueError(self.kind)
        self.clf.fit(X, Y[:, self.keep])
        self.train_seconds = time.perf_counter() - t
        return self

    def predict_proba(self, texts) -> np.ndarray:
        X = self._vec(list(texts))
        if self.kind == "mlp":
            P = self.clf.predict_proba(self.svd.transform(X))
        elif self.kind == "svm":
            P = _sigmoid(2.0 * self.clf.decision_function(X))          # Platt-like squashing; calibrated scores are NOT claimed for the SVM
        else:
            P = self.clf.predict_proba(X)
        out = np.zeros((len(texts), len(self.labels)))
        out[:, self.keep] = P
        return out

    def predict(self, texts, thr: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= thr).astype(int)

    def predict_one(self, text: str) -> tuple[str, float]:
        """Most probable category (or 'no_action') for a single span with its probability."""
        p = self.predict_proba([text])[0]
        j = int(np.argmax(p))
        return self.labels[j], float(p[j])

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ActionClassifier":
        return joblib.load(path)


def evaluate(model: ActionClassifier, df: pd.DataFrame, thr: float = 0.5) -> dict:
    Y = labels_matrix(df)
    P = model.predict(df.text.tolist(), thr)
    rep = multilabel_report(Y, P, ALL_LABELS)
    single = df[Y.sum(axis=1) == 1]
    ys = [ALL_LABELS[int(np.argmax(r))] for r in labels_matrix(single)]
    pr = model.predict_proba(single.text.tolist())
    yp = [ALL_LABELS[int(np.argmax(r))] for r in pr]
    rep["single_label_accuracy"] = float(np.mean([a == b for a, b in zip(ys, yp)])) if ys else float("nan")
    rep["confusion_single_label"] = single_label_confusion(ys, yp, ALL_LABELS)
    rep["n"] = int(len(df))
    return rep


def evaluate_on_spans(model: ActionClassifier, docs: list[dict]) -> dict:
    """Score the classifier on developer-annotated obligation spans (task_type per span) - independent of the silver labels."""
    spans = [(o["span"], o["task_type"]) for d in docs for o in d["obligations"]]
    texts = [s for s, _ in spans]
    y_true = [t for _, t in spans]
    pr = model.predict_proba(texts)
    y_pred = [model.labels[int(np.argmax(r))] for r in pr]
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    from sklearn.metrics import f1_score
    return {"n_spans": len(spans), "accuracy": acc, "macro_f1": float(f1_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0)),
            "confusion": single_label_confusion(y_true, y_pred, CATEGORIES + ["no_action"]), "source": "developer_annotated_challenge_set"}


def tune(train: pd.DataFrame, val: pd.DataFrame, kind: str, budget: int, seed: int = 0, extra_train: pd.DataFrame | None = None) -> tuple[ActionClassifier, pd.DataFrame]:
    """Random search on (train -> validation macro-F1). The test split is never touched here."""
    rng = np.random.default_rng(seed)
    tr = pd.concat([train, extra_train]) if extra_train is not None else train
    Ytr, Yva = labels_matrix(tr), labels_matrix(val)
    rows, best, best_f = [], None, -1
    space = {"C": [0.1, 0.3, 1.0, 3.0, 10.0], "word_ngrams": [(1, 1), (1, 2)], "char": [True, False], "class_weight": ["balanced", None]}
    for k in range(budget):
        cfgk = {n: v[int(rng.integers(len(v)))] for n, v in space.items()}
        if kind == "mlp":
            cfgk["C"] = 1.0
        m = ActionClassifier(kind, seed=seed, **cfgk).fit(tr.text, Ytr)
        P = m.predict(val.text.tolist())
        f = multilabel_report(Yva, P, ALL_LABELS)["macro"]["f1"]
        rows.append({**{k2: str(v) for k2, v in cfgk.items()}, "val_macro_f1": f, "train_s": m.train_seconds})
        if f > best_f:
            best, best_f = m, f
    return best, pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
