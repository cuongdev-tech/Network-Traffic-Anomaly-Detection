import tempfile
import unittest
import contextlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.tabnet_training import make_preprocessor, split_validation, train_tabnet
from src.tune_tabnet_unsw import improves_validation


class TabNetTrainingTests(unittest.TestCase):
    def test_two_epochs_save_reload_and_predict_through_application(self):
        from giao_dien.prediction import predict_file
        frame = pd.DataFrame({"dur": np.arange(2000, dtype=float), "proto": ["tcp", "udp"] * 1000})
        labels = np.tile([0, 1], 1000)
        with contextlib.redirect_stdout(io.StringIO()):
            model, preprocessor, metadata = train_tabnet(frame, labels, max_epochs=2, threads=1)
        self.assertEqual(metadata["epochs_run"], 2)
        self.assertEqual(metadata["fit_train"] + metadata["validation"], len(frame))
        self.assertEqual(model.input_dim, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "models" / "unsw_nb15"
            folder.mkdir(parents=True)
            with contextlib.redirect_stdout(io.StringIO()):
                model.save_model(str(folder / "tabnet_attack"))
            joblib.dump({"preprocessor": preprocessor, "classes": ["Generic", "Normal"],
                         "features": list(frame)}, folder / "tabnet_bundle.joblib")
            incoming = pd.DataFrame({"dur": [3., 10000.], "proto": ["tcp", "unseen"]})
            incoming.to_csv(root / "input.csv", index=False)
            result = predict_file(root, "unsw", "TabNet", root / "input.csv")
            self.assertEqual(len(result), 2)
            self.assertTrue(set(result.prediction_class) <= {"Bình thường", "Generic"})
            self.assertTrue(result.prediction_probability.between(0, 1).all())

    def test_preprocessor_reserves_unseen_category_and_survives_reload(self):
        training = pd.DataFrame({"duration": np.arange(1000, dtype=float),
                                 "proto": ["tcp", "udp"] * 500})
        preprocessor = make_preprocessor(training).fit(training)
        incoming = pd.DataFrame({"duration": [5.0, 1e12], "proto": ["tcp", "unseen"]})
        transformed = preprocessor.transform(incoming)
        self.assertEqual(transformed.shape, (2, 2))
        self.assertEqual(transformed[1, 1], 0)
        self.assertGreater(transformed[0, 1], 0)
        self.assertTrue(np.isfinite(transformed).all())
        # Validation/prediction must not change fitted numeric statistics or vocabularies.
        quantiles = preprocessor.named_transformers_["num"].named_steps["quantile"].quantiles_
        self.assertEqual(quantiles.max(), 999)
        categories = preprocessor.named_transformers_["cat"].named_steps["encoder"].categories_[0]
        self.assertNotIn("unseen", categories)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            joblib.dump(preprocessor, path)
            np.testing.assert_array_equal(joblib.load(path).transform(incoming), transformed)

    def test_validation_is_disjoint_stratified_and_reproducible(self):
        frame = pd.DataFrame({"x": np.arange(1000)})
        labels = np.tile(np.arange(10), 100)
        training, validation, yt, yv = split_validation(frame, labels)
        self.assertFalse(set(training.index) & set(validation.index))
        self.assertEqual(len(training) + len(validation), len(frame))
        np.testing.assert_array_equal(np.bincount(yv), np.full(10, 10))
        np.testing.assert_array_equal(validation.index, split_validation(frame, labels)[1].index)

    def test_promotion_requires_accuracy_gain_without_macro_f1_regression(self):
        incumbent = {"accuracy": .75, "f1_macro": .41}
        self.assertTrue(improves_validation({"accuracy": .82, "f1_macro": .60}, incumbent))
        self.assertFalse(improves_validation({"accuracy": .82, "f1_macro": .40}, incumbent))
        self.assertFalse(improves_validation({"accuracy": .74, "f1_macro": .60}, incumbent))


if __name__ == "__main__":
    unittest.main()
