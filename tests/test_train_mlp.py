import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_mlp


class MLPTests(unittest.TestCase):
    def test_preprocessing_and_model_load_in_a_fresh_process(self):
        numeric = pd.DataFrame({
            "duration": [-10.0, 0, 1, 2, np.nan, 1e9] * 6,
            "constant": [0.0] * 36,
        })
        for categorical in (False, True):
            with self.subTest(categorical=categorical), tempfile.TemporaryDirectory() as tmp:
                features = numeric.copy().fillna(0) if not categorical else numeric.copy()
                if categorical:
                    features["proto"] = ["tcp", "udp"] * 18
                model = train_mlp.build_model(features, categorical)
                model.set_params(classifier__hidden_layer_sizes=(4,),
                                 classifier__early_stopping=False,
                                 classifier__max_iter=2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    model.fit(features, np.array([0, 1] * 18))
                query = features.iloc[:2].fillna(0).copy()
                if categorical:
                    query["proto"] = "unseen-protocol"
                transformed = model[:-1].transform(query)
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                self.assertTrue(np.isfinite(transformed).all())
                model_path = Path(tmp) / "model.joblib"
                query_path = Path(tmp) / "query.json"
                joblib.dump(model, model_path)
                query.to_json(query_path, orient="split")
                result = subprocess.run(
                    [sys.executable, "-I", "-c",
                     "import joblib,pandas as pd,sys,json; "
                     "m=joblib.load(sys.argv[1]); "
                     "x=pd.read_json(sys.argv[2],orient='split'); "
                     "print(json.dumps(m.predict(x).tolist()))",
                     str(model_path), str(query_path)],
                    capture_output=True, text=True, check=True,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
                )
                self.assertEqual(json.loads(result.stdout), model.predict(query).tolist())

    def test_comparison_uses_only_the_current_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = [{"model": name, "dataset": "cic", "accuracy": 0.9,
                     "precision_weighted": 0.9, "recall_weighted": 0.9,
                     "f1_weighted": 0.9} for name in ("Random Forest", "XGBoost")]
            pd.DataFrame(rows).to_csv(
                directory / "random_forest_xgboost_metrics.csv", index=False)
            pd.DataFrame(rows + [{**rows[0], "dataset": "unsw"}]).to_csv(
                directory / "random_forest_xgboost_all_metrics.csv", index=False)
            train_mlp.save_comparison(directory, {**rows[0], "model": "MLP"})
            result = pd.read_csv(directory / "all_models_metrics.csv")
            self.assertEqual(result["dataset"].tolist(), ["cic"] * 3)
            self.assertEqual(set(result["model"]), {"Random Forest", "XGBoost", "MLP"})


if __name__ == "__main__":
    unittest.main()
