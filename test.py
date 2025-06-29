import unittest
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model

class SanityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Load the saved model once
        cls.model = load_model("model.keras")

        # 2. Load the iris dataset
        cls.df = pd.read_csv("iris.csv")

        # 3. Define expected columns
        cls.expected_cols = {
            "sepal_length", "sepal_width",
            "petal_length", "petal_width", "species"
        }

        # 4. Hard-coded test samples: (features, true_species)
        cls.samples = [
            ([5.1, 3.5, 1.4, 0.2], "setosa"),
            ([6.7, 3.1, 4.7, 1.5], "versicolor"),
            ([7.2, 3.6, 6.1, 2.5], "virginica"),
        ]

        # 5. Label mapping if your model outputs one-hot or integer indices
        cls.labels = ["setosa", "versicolor", "virginica"]

    def test_data_columns(self):
        """iris.csv must have all required columns."""
        self.assertTrue(
            self.expected_cols.issubset(set(self.df.columns)),
            f"Missing one of {self.expected_cols}"
        )

    def test_no_missing_values(self):
        """iris.csv must contain no nulls."""
        self.assertFalse(
            self.df.isnull().any().any(),
            "Found missing/null values in iris.csv"
        )

    def test_metrics_file_exists(self):
        """metrics.csv must exist and include an accuracy column."""
        self.assertTrue(os.path.exists("metrics.csv"), "metrics.csv not found")
        meta = pd.read_csv("metrics.csv")
        self.assertIn("accuracy", meta.columns, "metrics.csv missing 'accuracy'")

    def test_model_sample_predictions(self):
        """The saved model must correctly classify three known iris samples."""
        for features, true_label in self.samples:
            x = np.array([features])              # batch of size 1
            preds = self.model.predict(x)          # shape ➔ (1, 3)
            idx = np.argmax(preds, axis=1)[0]     # pick highest-score class
            pred_label = self.labels[idx]
            self.assertEqual(
                pred_label, true_label,
                f"Expected {true_label} for {features}, got {pred_label}"
            )

if __name__ == "__main__":
    # Run all tests and exit with appropriate code
    unittest.main(verbosity=2)
