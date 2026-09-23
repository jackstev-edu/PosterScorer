"""Continuous score feedback adapted from the supplied temp.py ScoreOptimizer.

This searches training quantiles using an already-trained predictor; it does not
train a second model or identify causal feature harm. Text/graphic area, text-count and text-height groups are actionable.
Geometry stays fixed; related measurements are updated as coherent groups.
Count/height edits assume text area scales proportionally; this is a model
probe, not an image edit or a guarantee of realizable design improvement.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

AREA_FEATURES = ("text_area_frac", "graphic_area_frac")
GROUPS = {"word_count": ("word_count", "char_count", "text_block_count"),
          "mean_text_height_frac": ("mean_text_height_frac", "max_text_height_frac", "text_height_std_frac")}
DERIVED = ("background_area_frac", "text_share", "text_to_graphic_ratio")
RATIO_EPS = 1e-3  # Same definition as features.extract_features.


class ScoreOptimizer:
    """Attach to a fitted predictor and learn candidate values from TRAIN only."""

    def __init__(self, predictor, train_features, feature_names=None, training_ids=None):
        self.predictor = predictor
        self.feature_names = list(feature_names or train_features.columns)
        train = self._validate(train_features)
        if train.empty:
            raise ValueError("Training features cannot be empty")
        self.actionable = list(AREA_FEATURES) + [name for name, group in GROUPS.items()
                                              if set(group).issubset(self.feature_names)]
        self.quantiles = {
            name: np.unique(train[name].quantile(np.linspace(0.05, 0.95, 19))).tolist()
            for name in self.actionable
        }
        self.recipe = {
            "version": 1, "method": "training_quantile_group_counterfactuals", "fit_split": "train",
            "training_rows": len(train), "feature_columns": self.feature_names,
            "actionable_features": self.actionable, "quantiles": self.quantiles,
            "coupled_groups": {name: list(group) + ["text_area_frac"] + list(DERIVED)
                               for name, group in GROUPS.items() if name in self.actionable},
            "immutable_features": [c for c in self.feature_names if c not in AREA_FEATURES + DERIVED + tuple(x for group in GROUPS.values() for x in group)],
            "interpretation": "Model-estimated layout changes, not causal attribution or feedback labels.",
        }
        if training_ids is not None:
            ids = sorted(str(value) for value in training_ids)
            if len(ids) != len(train) or len(set(ids)) != len(ids):
                raise ValueError("Training IDs must uniquely match training rows")
            self.recipe["training_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()

    def save_recipe(self, path):
        Path(path).write_text(json.dumps(self.recipe, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load_recipe(cls, predictor, path):
        """Reuse fitted candidate metadata without reading any dataset splits."""
        recipe = json.loads(Path(path).read_text())
        if recipe.get("fit_split") != "train" or recipe.get("version") != 1:
            raise ValueError("Expected a version 1 recipe fitted on train")
        obj = cls.__new__(cls)
        obj.predictor = predictor
        obj.recipe = recipe
        obj.feature_names = recipe["feature_columns"]
        obj.actionable = recipe["actionable_features"]
        obj.quantiles = recipe["quantiles"]
        if not set(obj.actionable).issubset(set(AREA_FEATURES) | set(GROUPS)):
            raise ValueError("Unknown actionable feature in recipe")
        for name in obj.actionable:
            values = np.asarray(obj.quantiles[name], dtype=float)
            if not len(values) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid candidate values in feedback recipe")
        return obj

    def _validate(self, frame):
        missing = set(self.feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        required_areas = set(AREA_FEATURES + DERIVED)
        if not required_areas.issubset(self.feature_names):
            raise ValueError("Feedback requires the five shared area features")
        values = frame[self.feature_names].apply(pd.to_numeric, errors="raise").astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Feedback requires finite, non-missing feature values")
        for column in values:
            if (values[column] < 0).any():
                raise ValueError(f"{column} cannot be negative")
            if (column.endswith("_frac") or column == "text_share") and (values[column] > 1).any():
                raise ValueError(f"{column} must be between 0 and 1")
        text, graphic = values[AREA_FEATURES[0]], values[AREA_FEATURES[1]]
        if (text + graphic > 1 + 1e-8).any():
            raise ValueError("Text and graphic areas cannot exceed the image area")
        expected = {
            "background_area_frac": np.maximum(0, 1 - text - graphic),
            "text_share": np.divide(text, text + graphic, out=np.zeros(len(text)), where=(text + graphic) > 0),
            "text_to_graphic_ratio": text / np.maximum(graphic, RATIO_EPS),
        }
        for column, expected_values in expected.items():
            if not np.allclose(values[column], expected_values, rtol=1e-5, atol=1e-7):
                raise ValueError(f"Inconsistent derived feature: {column}")
        return values.reset_index(drop=True)

    def candidate_rows(self, features):
        """Return feasible rows and (feature, old value, new value) descriptions."""
        base = self._validate(pd.DataFrame([features])).iloc[0]
        rows, descriptions = [], []
        for feature in self.actionable:
            if feature != "graphic_area_frac" and (base["text_area_frac"] == 0 or base.get("word_count", 1) == 0):
                continue
            for value in self.quantiles[feature]:
                if feature == "word_count":
                    value = float(round(value))
                if np.isclose(value, base[feature]):
                    continue
                row = base.copy()
                row[feature] = value
                if feature in GROUPS:
                    if value <= 0 or base[feature] <= 0:
                        continue
                    factor = value / base[feature]
                    if feature == "word_count":
                        row["char_count"] = max(value, round(base["char_count"] * factor))
                        row["text_block_count"] = min(value, max(1, round(base["text_block_count"] * factor)))
                    else:
                        row["max_text_height_frac"] = base["max_text_height_frac"] * factor
                        row["text_height_std_frac"] = base["text_height_std_frac"] * factor
                        if row["max_text_height_frac"] > 1 or row["text_height_std_frac"] > row["max_text_height_frac"] / 2 + 1e-8:
                            continue
                    row["text_area_frac"] = base["text_area_frac"] * factor
                text, graphic = row["text_area_frac"], row["graphic_area_frac"]
                if text < 0 or graphic < 0 or text + graphic > 1 or (text == 0 and base.get("word_count", 0) > 0):
                    continue
                row["background_area_frac"] = max(0.0, 1 - text - graphic)
                row["text_share"] = text / (text + graphic) if text + graphic > 0 else 0.0
                row["text_to_graphic_ratio"] = text / max(graphic, RATIO_EPS)
                changes = {name: float(row[name]) for name in self.feature_names if not np.isclose(row[name], base[name])}
                rows.append(row)
                descriptions.append((feature, float(base[feature]), float(value), changes))
        return pd.DataFrame(rows, columns=self.feature_names), descriptions

    def analyze_many(self, features, top_k=3, min_improvement=1e-6):
        """Score every input and all feasible candidates in one prediction batch."""
        if top_k < 1 or min_improvement < 0:
            raise ValueError("top_k must be positive and min_improvement nonnegative")
        base = self._validate(features)
        if base.empty:
            return []
        parts, metadata = [base], []
        for index, row in base.iterrows():
            candidates, descriptions = self.candidate_rows(row.to_dict())
            parts.append(candidates)
            metadata.extend((index, *description) for description in descriptions)
        scores = np.asarray(self.predictor.predict(pd.concat(parts, ignore_index=True)), dtype=float)
        if len(scores) != sum(len(part) for part in parts) or not np.isfinite(scores).all():
            raise ValueError("Predictor returned invalid scores")
        results = [{"score": float(score), "ranked_changes": [], "best_recommendation": None,
                    "best_feature": None, "best_gain": None, "best_direction": None,
                    "supported_features": self.actionable,
                    "feedback_sentence": "The model found no tested layout change likely to improve this poster’s score.",
                    "message": "No tested feature-group change improved the model's predicted score."}
                   for score in scores[:len(base)]]
        best_by_row = [{} for _ in range(len(base))]
        for (index, feature, old, new, changes), score in zip(metadata, scores[len(base):]):
            change = {"feature": feature, "current_value": old, "suggested_value": new,
                      "direction": "increase" if new > old else "decrease", "coupled_feature_updates": changes,
                      "predicted_score": float(score), "gain": float(score - results[index]["score"])}
            previous = best_by_row[index].get(feature)
            if previous is None or change["gain"] > previous["gain"]:
                best_by_row[index][feature] = change
        for result, best in zip(results, best_by_row):
            ranked = sorted(best.values(), key=lambda item: item["gain"], reverse=True)[:top_k]
            result["ranked_changes"] = ranked  # Signed gains include zero/negative outcomes.
            if ranked and ranked[0]["gain"] > min_improvement:
                recommendation = ranked[0]
                result.update(best_recommendation=recommendation, best_feature=recommendation["feature"],
                              best_gain=recommendation["gain"], best_direction=recommendation["direction"],
                              message="Largest estimated gain among tested feature-group changes; not a causal diagnosis.")
                labels = {"text_area_frac": "text coverage", "graphic_area_frac": "graphic coverage",
                          "word_count": "the amount of text", "mean_text_height_frac": "text size"}
                action = "increasing" if recommendation["direction"] == "increase" else "decreasing"
                result["feedback_sentence"] = (
                    f"The model suggests {action} {labels[recommendation['feature']]} "
                    "as the most promising change for this poster."
                )
        return results

    def analyze(self, features, **kwargs):
        return self.analyze_many(pd.DataFrame([features]), **kwargs)[0]
