import unittest
import pandas as pd
from math import isnan
import os
import tempfile
from src.unitxt.loaders import LoadCSV

class TestLoaders(unittest.TestCase):

    def test_load_csv(self):
        # Using a context for the temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:

            files = {}
            dfs = {}

            for file in ['train', 'test']:
                path = os.path.join(tmp_dir, file + '.csv')  # Adding a file extension
                df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})  # Replace with your data
                dfs[file] = df
                df.to_csv(path, index=False)
                files[file] = path

            loader = LoadCSV(files=files)
            ms = loader()

            for file in ['train', 'test']:
                for saved_instance, loaded_instance in zip(dfs[file].iterrows(), ms[file]):
                    self.assertEqual(saved_instance[1].to_dict(), loaded_instance)

