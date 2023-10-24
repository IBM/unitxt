import os
import tempfile
import unittest
from math import isnan

import pandas as pd
from src.unitxt.loaders import LoadCSV, LoadFromIBMCloud


class TestLoaders(unittest.TestCase):
    def test_load_csv(self):
        # Using a context for the temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = {}
            dfs = {}

            for file in ["train", "test"]:
                path = os.path.join(tmp_dir, file + ".csv")  # Adding a file extension
                df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})  # Replace with your data
                dfs[file] = df
                df.to_csv(path, index=False)
                files[file] = path

            loader = LoadCSV(files=files)
            ms = loader()

            for file in ["train", "test"]:
                for saved_instance, loaded_instance in zip(dfs[file].iterrows(), ms[file]):
                    self.assertEqual(saved_instance[1].to_dict(), loaded_instance)

    def test_load_from_ibm_cos(self):
        os.environ["DUMMY_URL_ENV"] = "DUMMY_URL"
        os.environ["DUMMY_KEY_ENV"] = "DUMMY_KEY"
        os.environ["DUMMY_SECRET_ENV"] = "DUMMY_SECRET"
        loader = LoadFromIBMCloud(
            endpoint_url_env="DUMMY_URL_ENV",
            aws_access_key_id_env="DUMMY_KEY_ENV",
            aws_secret_access_key_env="DUMMY_SECRET_ENV",
            bucket_name="DUMMY_BUCKET",
            data_dir="DUMMY_DATA_DIR",
            data_files="DUMMY_DATA_FILES",
        )
