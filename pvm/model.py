"""The predictive-maintenance model.

What it learns
--------------
For every failure mode (target column, e.g. "fail_brakes") it trains a
gradient-boosted decision-tree classifier that outputs the probability that
component fails within the prediction horizon. It also learns:

* remaining useful life (km until the next required repair), with an 80 %
  prediction interval from two quantile regressors, and
* an anomaly detector (Isolation Forest) trained only on healthy vehicles,
  which flags readings that look unlike anything healthy — useful for
  failure types the labels never covered.

Why gradient-boosted trees (HistGradientBoosting)?
--------------------------------------------------
* They are the state of the art for tabular sensor data (usually beating
  neural nets on this kind of table).
* They handle missing values natively: if a car has no brake-pad sensor, the
  tree learns which branch "unknown" should go down. That is what lets one
  model serve EVs, diesels and cars with different sensor packages.
* They handle categorical inputs (vehicle type, climate) directly.
* Boosting = many small trees, each one trained to fix the mistakes of the
  trees before it.

Nothing in this file is specific to cars: give `MaintenanceModel` any
table plus the names of its label columns and it will train on every other
column. That is how it "adapts to different parameters".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from .features import add_derived_features

MAX_CATEGORIES = 250


@dataclass
class MaintenanceModel:
    classification_targets: list[str]
    regression_target: str | None = None
    applicability: dict[str, tuple[str, set]] = field(default_factory=dict)
    group_col: str | None = None          # baselines for explanations are computed per group (e.g. powertrain)
    drop_cols: list[str] = field(default_factory=list)
    random_state: int = 0

    # ------------------------------------------------------------------
    # Preparing inputs
    # ------------------------------------------------------------------
    def _infer_columns(self, df: pd.DataFrame) -> None:
        """Decide which columns are inputs and whether each is numeric or categorical."""
        excluded = set(self.classification_targets) | set(self.drop_cols)
        if self.regression_target:
            excluded.add(self.regression_target)
        self.raw_numeric_, self.raw_categorical_ = [], []
        for col in df.columns:
            if col in excluded or col.startswith("_"):
                continue  # "_" columns are hidden simulator truth — never a model input
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                self.raw_numeric_.append(col)
            elif pd.api.types.is_numeric_dtype(s):
                self.raw_numeric_.append(col)
            else:
                as_num = pd.to_numeric(s, errors="coerce")
                if as_num.notna().sum() >= 0.95 * s.notna().sum() and s.notna().any():
                    self.raw_numeric_.append(col)
                elif s.nunique() <= MAX_CATEGORIES and s.nunique() < 0.5 * len(s):
                    self.raw_categorical_.append(col)
                # else: free text or an ID column — not useful, skip it
        self.categories_ = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.raw_categorical_}

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw table -> model matrix. Missing columns become NaN, unknown categories become NaN."""
        d = pd.DataFrame(index=df.index)
        for c in self.raw_numeric_:
            d[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else np.nan
        for c in self.raw_categorical_:
            vals = df[c].astype(str).where(df[c].notna()) if c in df else pd.Series(np.nan, index=df.index)
            vals = vals.where(vals.isin(self.categories_[c]))  # unseen category -> missing
            d[c] = pd.Categorical(vals, categories=self.categories_[c])
        d = add_derived_features(d)
        return d[self.feature_names_] if hasattr(self, "feature_names_") else d

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, test_size: float = 0.2, compute_importance: bool = True,
            progress=None) -> dict:
        """Train every sub-model and return a metrics report measured on held-out data."""
        df = df.reset_index(drop=True)
        self._infer_columns(df)
        X_all = self._prepare(df)
        self.feature_names_ = list(X_all.columns)
        self.derived_features_ = [c for c in self.feature_names_ if c not in self.raw_numeric_ + self.raw_categorical_]

        # One split shared by every target so the test set is truly unseen.
        # Stratify by group (e.g. powertrain) when we have one.
        strat = df[self.group_col] if self.group_col in df else None
        idx_train, idx_test = train_test_split(df.index, test_size=test_size,
                                               random_state=self.random_state, stratify=strat)
        idx_train, idx_val = train_test_split(idx_train, test_size=0.15, random_state=self.random_state)

        self.classifiers_, self.thresholds_, self.importances_, self.target_features_ = {}, {}, {}, {}
        report = {"n_rows": len(df), "n_features": len(self.feature_names_), "targets": {}}
        steps = len(self.classification_targets) + 2
        for i, target in enumerate(self.classification_targets):
            if progress:
                progress(i / steps, f"Training {target}")
            y = pd.to_numeric(df[target], errors="coerce")
            tr, va, te = (ix[y.loc[ix].notna().to_numpy()] for ix in (idx_train, idx_val, idx_test))
            if y.loc[tr].nunique() < 2:
                report["targets"][target] = {"skipped": "only one class present"}
                continue

            clf = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=30,
                l2_regularization=0.1, categorical_features="from_dtype",
                early_stopping=True, validation_fraction=0.12, n_iter_no_change=25,
                random_state=self.random_state)
            # Each target only uses columns that have data for its vehicles
            # (e.g. EV-only sensors are blank for every car that has engine oil).
            cols = [c for c in self.feature_names_ if X_all.loc[tr, c].notna().any()]
            self.target_features_[target] = cols
            clf.fit(X_all.loc[tr, cols], y.loc[tr].astype(int))
            self.classifiers_[target] = clf

            # Pick the alert threshold on the VALIDATION set (maximise F1),
            # then score on the TEST set — never tune on the data you grade on.
            p_val = clf.predict_proba(X_all.loc[va, cols])[:, 1]
            self.thresholds_[target] = _best_f1_threshold(y.loc[va].astype(int).to_numpy(), p_val)

            p_te = clf.predict_proba(X_all.loc[te, cols])[:, 1]
            y_te = y.loc[te].astype(int).to_numpy()
            pred = (p_te >= self.thresholds_[target]).astype(int)
            report["targets"][target] = {
                "n_test": int(len(te)),
                "prevalence": float(y_te.mean()),
                "roc_auc": float(roc_auc_score(y_te, p_te)) if len(set(y_te)) > 1 else None,
                "pr_auc": float(average_precision_score(y_te, p_te)) if len(set(y_te)) > 1 else None,
                "brier": float(brier_score_loss(y_te, p_te)),
                "threshold": float(self.thresholds_[target]),
                "precision": float(precision_score(y_te, pred, zero_division=0)),
                "recall": float(recall_score(y_te, pred, zero_division=0)),
                "f1": float(f1_score(y_te, pred, zero_division=0)),
                "n_trees": int(clf.n_iter_),
            }
            if compute_importance and len(set(y_te)) > 1:
                sample = te[: min(len(te), 2500)]
                pi = permutation_importance(clf, X_all.loc[sample, cols], y.loc[sample].astype(int),
                                            scoring="roc_auc", n_repeats=3, random_state=self.random_state)
                self.importances_[target] = (pd.Series(pi.importances_mean, index=cols)
                                             .sort_values(ascending=False))

        # ---- Remaining useful life: median + 10th/90th percentile models ----
        if self.regression_target and self.regression_target in df:
            if progress:
                progress((steps - 2) / steps, "Training remaining-life model")
            y = pd.to_numeric(df[self.regression_target], errors="coerce")
            tr = idx_train[y.loc[idx_train].notna().to_numpy()]
            te = idx_test[y.loc[idx_test].notna().to_numpy()]
            common = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=30,
                          categorical_features="from_dtype", early_stopping=True, random_state=self.random_state)
            self.regressors_ = {
                q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **common).fit(X_all.loc[tr], y.loc[tr])
                for q in (0.1, 0.5, 0.9)
            }
            pred = {q: m.predict(X_all.loc[te]) for q, m in self.regressors_.items()}
            y_te = y.loc[te].to_numpy()
            report["regression"] = {
                "target": self.regression_target,
                "mae": float(mean_absolute_error(y_te, pred[0.5])),
                "r2": float(r2_score(y_te, pred[0.5])),
                "interval_coverage": float(np.mean((y_te >= pred[0.1]) & (y_te <= pred[0.9]))),
                "target_std": float(np.std(y_te)),
            }
        else:
            self.regressors_ = {}

        # ---- Anomaly detector trained on healthy vehicles only ----
        if progress:
            progress((steps - 1) / steps, "Training anomaly detector")
        labels = df[self.classification_targets].apply(pd.to_numeric, errors="coerce").fillna(0)
        healthy = idx_train[(labels.loc[idx_train].max(axis=1) == 0).to_numpy()]
        num_cols = [c for c in self.feature_names_ if c not in self.raw_categorical_]
        self.anomaly_cols_ = [c for c in num_cols if X_all.loc[healthy, c].notna().any()]
        self.anomaly_ = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            IsolationForest(n_estimators=300, contamination="auto", random_state=self.random_state),
        ).fit(X_all.loc[healthy, self.anomaly_cols_])
        self.healthy_scores_ = np.sort(-self.anomaly_.score_samples(X_all.loc[healthy, self.anomaly_cols_]))

        # ---- Baselines for explanations: typical healthy values, per group ----
        raw_healthy = df.loc[healthy]
        self.baselines_ = {"__all__": raw_healthy[self.raw_numeric_].median(numeric_only=True)}
        if self.group_col in df:
            for g, sub in raw_healthy.groupby(self.group_col):
                self.baselines_[str(g)] = sub[self.raw_numeric_].median(numeric_only=True)

        self.report_ = report
        if progress:
            progress(1.0, "Done")
        return report

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _applicable(self, df: pd.DataFrame, target: str) -> np.ndarray:
        rule = self.applicability.get(target)
        if not rule or rule[0] not in df:
            return np.ones(len(df), dtype=bool)
        col, allowed = rule
        return df[col].astype(str).isin(allowed).to_numpy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one column per output: risk_<target> (0–1, NaN = not applicable),
        rul_p10/p50/p90, anomaly_pct (0–100) and overall_risk."""
        X = self._prepare(df)
        out = pd.DataFrame(index=df.index)
        for target, clf in self.classifiers_.items():
            p = clf.predict_proba(X[self.target_features_[target]])[:, 1]
            out[f"risk_{target}"] = np.where(self._applicable(df, target), p, np.nan)
        risks = out.filter(like="risk_")
        # P(at least one failure), assuming independence: 1 - Π(1 - p_i)
        out["overall_risk"] = 1 - (1 - risks.fillna(0)).prod(axis=1)
        for q, m in self.regressors_.items():
            out[f"rul_p{int(q * 100)}"] = np.clip(m.predict(X), 0, None)
        score = -self.anomaly_.score_samples(X[self.anomaly_cols_])
        out["anomaly_pct"] = np.searchsorted(self.healthy_scores_, score) / len(self.healthy_scores_) * 100
        return out

    def explain(self, row: pd.DataFrame, target: str, top_k: int = 8) -> pd.DataFrame:
        """Why is this vehicle's risk what it is?

        Method (an "occlusion" / what-if explanation): for each input, swap
        this vehicle's value for the value a typical *healthy* vehicle of the
        same kind has, and see how much the predicted risk changes. A large
        drop means that input is pushing the risk up.
        """
        row = row.iloc[[0]]
        clf = self.classifiers_[target]
        cols = self.target_features_[target]
        p0 = clf.predict_proba(self._prepare(row)[cols])[0, 1]
        group = str(row[self.group_col].iloc[0]) if self.group_col in row else "__all__"
        base = self.baselines_.get(group, self.baselines_["__all__"])

        variants, names = [], []
        for c in self.raw_numeric_:
            if c not in row or pd.isna(row[c].iloc[0]) or c not in base or pd.isna(base[c]):
                continue
            r = row.copy()
            r[c] = base[c]
            variants.append(r)
            names.append(c)
        if not variants:
            return pd.DataFrame(columns=["feature", "value", "typical_healthy", "contribution"])
        p = clf.predict_proba(self._prepare(pd.concat(variants, ignore_index=True))[cols])[:, 1]
        res = pd.DataFrame({
            "feature": names,
            "value": [row[c].iloc[0] for c in names],
            "typical_healthy": [base[c] for c in names],
            "contribution": p0 - p,
        })
        return res.reindex(res["contribution"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "MaintenanceModel":
        return joblib.load(path)


def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return 0.5
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
