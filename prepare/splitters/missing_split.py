from src.unitxt.splitters import SplitRandomMix, SliceSplit
from src.unitxt import add_to_catalog

add_to_catalog(SplitRandomMix({'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'validation'}),
               'splitters.small_no_test', overwrite=True)
add_to_catalog(SliceSplit({'train': 'train[-500]', 'validation': 'train[500]', 'test': 'validation'}),
               'splitters.large_no_test', overwrite=True)
add_to_catalog(SplitRandomMix({'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'test'}),
               'splitters.small_no_dev', overwrite=True)
add_to_catalog(SliceSplit({'train': 'train[-500]', 'validation': 'train[500]', 'test': 'test'}),
               'splitters.large_no_dev', overwrite=True)
add_to_catalog(SliceSplit({'train': 'train[0]', 'validation': 'train[0]', 'test': 'test'}),
               'splitters.test_only', overwrite=True)
