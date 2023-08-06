from src.unitxt.test_utils.catalog import register_local_catalog_for_tests


def setUpClass(cls):
    register_local_catalog_for_tests()


def setup_unitxt_test_env(cls):
    setattr(cls, setUpClass.__name__, classmethod(setUpClass))
    return cls
