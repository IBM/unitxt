from datasets import load_dataset
from evaluate import load
from datasets.utils.py_utils import get_imports

import unittest
from src import unitxt
from src.unitxt.hf_utils, get_missing_imports
from src.unitxt.file_utils import get_all_files_in_dir
from pathlib import Path

class HFTests(unittest.TestCase):
    
    def test_dataset_imports(self):

        missing_imports = get_missing_imports(unitxt.dataset_file, exclude=['dataset', '__init__'])
        self.assertEqual(missing_imports, [])
        
    def test_metric_imports(self):
        
        missing_imports = get_missing_imports(unitxt.metric_file, exclude=['metric', '__init__', 'dataset'])
        self.assertEqual(missing_imports, [])
    
    def test_dataset_load(self):
        dataset = load_dataset(unitxt.dataset_file, 'card=cards.wnli,template_item=0', download_mode='force_redownload')
        
    
    def test_metric_load(self):
        metric = load(unitxt.metric_file)
        
        