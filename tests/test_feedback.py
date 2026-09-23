import unittest
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from score_feedback import ScoreOptimizer
from poster_inference import _extract_bundle


def row(text=0.2, graphic=0.4):
    return {"img_width_px": 1000, "img_height_px": 1000, "aspect_ratio": 1,
            "word_count": 10, "char_count": 50, "text_block_count": 2,
            "text_area_frac": text, "graphic_area_frac": graphic,
            "background_area_frac": 1-text-graphic,
            "text_share": text/(text+graphic) if text+graphic else 0,
            "text_to_graphic_ratio": text/max(graphic, 1e-3),
            "mean_text_height_frac": .03, "max_text_height_frac": .04,
            "text_height_std_frac": .01}


class Predictor:
    def __init__(self, constant=False):
        self.constant = constant
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return np.full(len(frame), 9.) if self.constant else 8 + frame.text_area_frac - frame.graphic_area_frac


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame([row(.1,.2), row(.3,.5), row(.4,.6)])
        self.train.loc[1, ["word_count","char_count","text_block_count"]] = [20,100,4]
        self.train.loc[1, ["mean_text_height_frac","max_text_height_frac","text_height_std_frac"]] = [.06,.08,.02]
        self.predictor = Predictor()
        self.optimizer = ScoreOptimizer(self.predictor, self.train, training_ids=['a','b','c'])

    def test_constraints_and_derived_areas(self):
        candidates, metadata = self.optimizer.candidate_rows(row(.2,.7))
        self.assertTrue(metadata)
        self.assertTrue(((candidates.text_area_frac+candidates.graphic_area_frac)<=1).all())
        np.testing.assert_allclose(candidates.background_area_frac, 1-candidates.text_area_frac-candidates.graphic_area_frac)
        np.testing.assert_allclose(candidates.text_share, candidates.text_area_frac/(candidates.text_area_frac+candidates.graphic_area_frac))
        np.testing.assert_allclose(candidates.text_to_graphic_ratio, candidates.text_area_frac/np.maximum(candidates.graphic_area_frac,1e-3))
        for field in ['word_count','char_count','text_block_count','aspect_ratio','img_width_px','mean_text_height_frac']:
            area_rows = candidates.iloc[[i for i, item in enumerate(metadata) if item[0] in ("text_area_frac","graphic_area_frac")]]
            self.assertTrue((area_rows[field]==row()[field]).all())

    def test_coupled_counts_and_height_constraints(self):
        candidates, metadata = self.optimizer.candidate_rows(row())
        self.assertTrue(any(item[0] == "word_count" for item in metadata))
        self.assertTrue(any(item[0] == "mean_text_height_frac" for item in metadata))
        self.assertTrue((candidates.char_count >= candidates.word_count).all())
        self.assertTrue((candidates.text_block_count <= candidates.word_count).all())
        self.assertTrue((candidates.word_count % 1 == 0).all())
        self.assertTrue((candidates.mean_text_height_frac <= candidates.max_text_height_frac).all())
        self.assertTrue((candidates.text_height_std_frac <= candidates.max_text_height_frac / 2).all())
        self.assertTrue((candidates.max_text_height_frac <= 1).all())

    def test_continuous_high_score_and_batched_predictions(self):
        results = self.optimizer.analyze_many(pd.DataFrame([row(),row(.3,.5)]))
        self.assertEqual(self.predictor.calls,1)
        self.assertGreater(results[0]['score'],7)
        self.assertGreater(results[0]['best_gain'],0)
        self.assertIn(results[0]['best_feature'],self.optimizer.actionable)

    def test_no_gain_and_signed_changes(self):
        optimizer = ScoreOptimizer(Predictor(constant=True),self.train)
        result = optimizer.analyze(row())
        self.assertIsNone(result['best_feature'])
        self.assertIsNone(result['best_recommendation'])
        self.assertIn('no tested', result['feedback_sentence'])
        self.assertTrue(all(change['gain']==0 for change in result['ranked_changes']))

    def test_train_only_recipe(self):
        self.assertEqual(self.optimizer.recipe['fit_split'],'train')
        self.assertEqual(self.optimizer.recipe['training_rows'],3)
        self.assertLessEqual(max(self.optimizer.quantiles['text_area_frac']),.4)
        self.assertNotIn('test',self.optimizer.recipe)
        self.assertIn('training_ids_sha256',self.optimizer.recipe)

    def test_feedback_sentence_matches_direction(self):
        result = self.optimizer.analyze(row())
        action = "increasing" if result["best_direction"] == "increase" else "decreasing"
        self.assertIn(action, result["feedback_sentence"])
        self.assertTrue(result["feedback_sentence"].startswith("The model suggests"))

    def test_recipe_reload_without_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feedback_recipe.json"
            self.optimizer.save_recipe(path)
            loaded = ScoreOptimizer.load_recipe(self.predictor, path)
            self.assertEqual(loaded.analyze(row()), self.optimizer.analyze(row()))

    def test_missing_and_incoherent_inputs(self):
        bad=row(); del bad['word_count']
        with self.assertRaises(ValueError): self.optimizer.analyze(bad)
        bad=row();bad['text_share']=.9
        with self.assertRaises(ValueError): self.optimizer.analyze(bad)
        bad=row();bad['char_count']=np.nan
        with self.assertRaises(ValueError): self.optimizer.analyze(bad)

    def test_zero_text_is_not_added_without_words(self):
        no_text=row(0,.5);no_text['word_count']=0
        _,metadata=self.optimizer.candidate_rows(no_text)
        self.assertTrue(all(item[0]!='text_area_frac' for item in metadata))


class BundleTests(unittest.TestCase):
    def test_safe_extract_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "bundle.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                for name in ("metrics.json", "feedback_recipe.json", "model_manifest.json", "model/predictor.pkl"):
                    archive.writestr(name, "placeholder")
            extracted = _extract_bundle(bundle, Path(folder) / "cache")
            self.assertTrue((extracted / "model/predictor.pkl").exists())
            self.assertEqual(extracted, _extract_bundle(bundle, Path(folder) / "cache"))

    def test_rejects_zip_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "bundle.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                for name in ("metrics.json", "feedback_recipe.json", "model_manifest.json", "model/predictor.pkl", "../escape"):
                    archive.writestr(name, "placeholder")
            with self.assertRaises(ValueError):
                _extract_bundle(bundle, Path(folder) / "cache")
            self.assertFalse((Path(folder) / "escape").exists())


if __name__ == '__main__':
    unittest.main()
